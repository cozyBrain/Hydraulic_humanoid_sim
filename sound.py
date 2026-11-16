"""Real-time robot sound for viewer.py (demo / testing only, the trainer never uses it).

Synthesised from the simulation state, so pitch and loudness follow the physics. Instead of plain oscillators
(which sound like an old game console) every source is an excitation driving resonances, the way machines sound:
  - gear pump       : one impulse per tooth mesh (pump rpm / 60 x 17 teeth), each slightly different, shaft-order
                      modulation, ringing through the housing's modes; pressure ripple + flow noise grow with pressure
  - EMRAX motor     : electrical frequency (rpm / 60 x pole pairs) harmonics with magnetic hum, bearing rumble
  - inverter        : spread-spectrum PWM hiss around the switching frequency, side bands at 2 x the electrical frequency
  - servo valves    : turbulent flow noise through a resonance whose pitch rises with the flow; clicks + knocks when
                      the flow changes suddenly (valves snapping open / shut, pressure hammer in the lines)
  - feet            : a thud + metal clank when a foot lands (foot contact forces from the viewer)
  - relief valve    : unstable, shrieking narrow-band noise when the supply pressure reaches the relief setting
  - room            : short stereo room reverb, bus compressor, soft limiter
Slow motion (G key) slows the sound with the simulation: frequencies are divided by the slow-down factor.

needs:  pip install sounddevice    and on Linux Mint:  sudo apt install libportaudio2
offline A/B:  python sound.py --demo out.wav      (a 12 s scripted run: spin-up, walking, relief, power off)
"""
import sys, threading, numpy as np
from scipy.signal import butter, lfilter

FS = 44100
PUMP_TEETH = 17          # pinion of the internal gear pump (robot24.html Z1, viz_util ratio 17/23)
POLE_PAIRS = 10          # EMRAX axial-flux rotor (assumed 10 pole pairs)
F_PWM = 8000.0           # inverter switching frequency (emDrive: 4..16 kHz configurable)
P_RELIEF = 225.0         # bar, screech starts just below the 230 bar relief
PUMP_MODES = ((640, 9, 1.0), (1190, 14, 0.8), (2080, 18, 0.6), (3350, 22, 0.42), (5150, 26, 0.25))  # housing (Hz, Q, gain)
CLANK_MODES = ((870, 25, 1.0), (2310, 35, 0.6), (4100, 40, 0.3))                                    # foot / frame metal


def _bp(f, q, fs=FS):
    """constant-peak band-pass biquad (RBJ), f clamped to the audio band"""
    f = float(np.clip(f, 20.0, fs * 0.45)); w = 2 * np.pi * f / fs; al = np.sin(w) / (2 * q)
    b = np.array([al, 0.0, -al]); a = np.array([1 + al, -2 * np.cos(w), 1 - al])
    return b / a[0], a / a[0]


class _Filt:
    """a filter that keeps its state across blocks (coefficients may change between blocks)"""
    def __init__(self, b, a):
        self.b, self.a = b, a; self.z = np.zeros(max(len(a), len(b)) - 1)

    def set(self, b, a): self.b, self.a = b, a

    def __call__(self, x):
        y, self.z = lfilter(self.b, self.a, x, zi=self.z); return y


class Synth:
    """block synthesiser: block(n) -> (n, 2) float32 stereo; set .target from any thread"""
    def __init__(self, fs=FS):
        self.fs = fs
        self.target = dict(rpm=0.0, Ps=0.0, Pel=0.0, Q=0.0, slow=1.0, on=1.0, gain=1.0, fl=0.0, fr=0.0)
        self.cur = dict(self.target)
        self.rng = np.random.default_rng(0)
        self.ph = np.zeros(16)                                       # oscillator phases
        self.mesh_ph = 0.0                                           # tooth-mesh impulse train phase (in teeth)
        self.pump_modes = [(_Filt(*_bp(f, q)), f, q, g) for f, q, g in PUMP_MODES]
        self.clank = [(_Filt(*_bp(f, q)), f, q, g) for f, q, g in CLANK_MODES]
        self.valve = _Filt(*_bp(2500, 2.5)); self.valve_f = 2500.0
        self.pwm = _Filt(*_bp(F_PWM, 18))
        self.relief = _Filt(*_bp(2900, 30))
        self.flow_lp = _Filt(*butter(2, 1800 / (fs / 2), 'low'))
        self.rumble_lp = _Filt(*butter(2, 160 / (fs / 2), 'low'))
        self.click_hp = _Filt(*butter(2, 3000 / (fs / 2), 'high'))
        self.thud_lp = _Filt(*butter(2, 700 / (fs / 2), 'low'))
        self.thump = _Filt(*_bp(75, 4))                              # low knock: pressure hammer / foot impact body
        self.pending = []                                            # [start sample, kind, amplitude] events
        self.prev_q = 0.0; self.prev_f = [0.0, 0.0]
        self.t = 0; self.last_step = [-10 ** 9, -10 ** 9]                # samples (a heel-toe bounce is one step)
        self.env = 0.0                                               # compressor envelope
        # stereo room: two decorrelated exponentially decaying noise tails, darker over time
        L = int(0.38 * fs); t = np.arange(L) / fs
        ir = []
        for ch in range(2):
            r = np.random.default_rng(10 + ch).standard_normal(L) * np.exp(-t / 0.085)
            r = lfilter(*butter(1, 5000 / (fs / 2), 'low'), r); r[: int(0.007 * fs)] = 0.0   # 7 ms pre-delay
            ir.append(r / np.sqrt(np.sum(r * r)))
        self.ir = np.array(ir)

    # ---- helpers
    def _osc(self, k, f, n):
        ph = self.ph[k] + 2 * np.pi * np.cumsum(f) / self.fs
        self.ph[k] = ph[-1] % (2 * np.pi)
        return np.sin(ph)

    def _reverb(self, x):
        """uniformly partitioned convolution (partitions of the block size): ~10x cheaper than one long FFT, which
        matters because the audio callback shares the Python interpreter with the simulation"""
        n = len(x)
        if getattr(self, '_rv_n', None) != n:              # (re)build the partitions for this block size
            L = self.ir.shape[1]; P = -(-L // n); self._rv_n = n
            irp = np.zeros((2, P * n)); irp[:, :L] = self.ir
            self._rv_H = np.fft.rfft(irp.reshape(2, P, n), 2 * n, axis=2)          # (2, P, n+1)
            self._rv_X = np.zeros((P, n + 1), complex); self._rv_prev = np.zeros(n); self._rv_i = 0
        self._rv_i = (self._rv_i - 1) % self._rv_X.shape[0]
        self._rv_X[self._rv_i] = np.fft.rfft(np.concatenate((self._rv_prev, x)))
        self._rv_prev = x.copy()
        X = np.roll(self._rv_X, -self._rv_i, axis=0)        # newest first, matching partition 0, 1, ...
        Y = np.einsum('pk,cpk->ck', X, self._rv_H)
        return np.fft.irfft(Y, 2 * n, axis=1)[:, n:]

    def _events(self, n):
        """one-shot sounds (valve clicks, pressure knocks, foot impacts) rendered into this block"""
        exc_click = np.zeros(n); exc_thump = np.zeros(n); exc_clank = np.zeros(n)
        keep = []
        for s0, kind, amp in self.pending:
            if s0 >= n: keep.append([s0 - n, kind, amp]); continue
            if kind == 'click':
                m = min(n - s0, 90); exc_click[s0:s0 + m] += amp * self.rng.standard_normal(m) * np.exp(-np.arange(m) / 18)
                exc_thump[s0] += amp * 0.6
            elif kind == 'step':
                m = min(n - s0, 600); exc_thump[s0] += amp * 2.2
                exc_clank[s0:s0 + m] += amp * self.rng.standard_normal(m) * np.exp(-np.arange(m) / 60)
        self.pending = keep
        click = self.click_hp(exc_click)
        thud = self.thump(exc_thump) * 6.0 + self.thud_lp(exc_clank) * 0.8
        clank = sum(g * flt(exc_clank) for flt, f, q, g in self.clank) * 0.5
        return click, thud, clank

    # ---- one block
    def block(self, n):
        t, c = self.target, self.cur
        a = np.linspace(0, 1, n); ramp = {}
        for k in c:
            ramp[k] = c[k] + (t[k] - c[k]) * a
            c[k] = t[k]
        slow = max(float(ramp['slow'][-1]), 1.0); on = ramp['on']
        rps = ramp['rpm'] / 60.0 / slow
        Ps = np.clip(ramp['Ps'], 0, 400); pres = Ps / 207.0
        load = np.clip(ramp['Pel'] / 15000.0, 0, 1.5)
        run = np.clip(ramp['rpm'] / 600.0, 0, 1)                   # 0 while the pump stands still
        noise = self.rng.standard_normal((4, n))

        # gear pump: one impulse per tooth mesh, jittered, ringing through the housing modes
        fm = rps * PUMP_TEETH
        teeth = self.mesh_ph + np.cumsum(fm) / self.fs
        idx = np.nonzero(np.diff(np.floor(np.concatenate(([self.mesh_ph], teeth)))))[0]
        self.mesh_ph = teeth[-1] % 1e6
        exc = np.zeros(n)
        if len(idx):
            shaft = 1 + 0.18 * np.sin(2 * np.pi * teeth[idx] / PUMP_TEETH) + 0.08 * np.sin(4 * np.pi * teeth[idx] / PUMP_TEETH)
            exc[idx] = shaft * np.exp(0.12 * self.rng.standard_normal(len(idx)))
        pump_modes = 0.0
        for flt, f, q, g in self.pump_modes:
            flt.set(*_bp(f / slow if slow > 1.5 else f, q)); pump_modes = pump_modes + g * flt(exc)
        ripple = 0.5 * self._osc(0, fm, n) + 0.25 * self._osc(1, 2 * fm, n)            # pressure ripple (low end)
        flow = self.flow_lp(noise[0]) * 2.2                                              # oil flow through the pump
        pump = run * (0.25 + 0.75 * np.clip(pres, 0, 1.3)) * (0.55 * pump_modes + 0.10 * ripple * pres + 0.05 * flow * pres)

        # motor: electrical harmonics (slightly unsteady), magnetic hum, bearing rumble
        fe = rps * POLE_PAIRS
        wob = 1 + 0.002 * self._osc(2, np.full(n, 3.1), n)
        motor = (0.5 * self._osc(3, fe * wob, n) + 0.25 * self._osc(4, 2 * fe, n) + 0.35 * self._osc(5, 6 * fe * wob, n)
                 + 0.12 * self._osc(6, 12 * fe, n)) * (0.015 + 0.06 * load) * run
        rumble = self.rumble_lp(noise[1]) * 0.6 * run * (0.4 + 0.6 * np.clip(rps * slow / 60, 0, 1))

        # inverter: narrow spread-spectrum hiss at the switching frequency, side bands at 2 fe (electronics: not slowed)
        pwm = self.pwm(noise[2]) * (1 + 0.6 * self._osc(7, 2 * fe * slow, n)) * 0.05 * np.clip(load * 2, 0, 1) * (slow < 1.5)

        # servo valves: turbulent flow, pitch rises with the flow; sudden flow changes -> click + knock
        q = max(float(ramp['Q'][-1]), 0.0)
        fv = (1400 + 2600 * np.sqrt(min(q, 60) / 20.0)) / slow
        self.valve_f += 0.3 * (fv - self.valve_f); self.valve.set(*_bp(self.valve_f, 2.2))
        hiss = self.valve(noise[3]) * np.clip(0.16 * np.sqrt(q / 20.0) * np.sqrt(np.clip(pres, 0, 1.5)), 0, 0.35)
        dq = q - self.prev_q; self.prev_q = q
        if abs(dq) > 2.5 and np.mean(pres) > 0.2:
            self.pending.append([int(self.rng.integers(0, n)), 'click', min(abs(dq) / 15.0, 1.0) * 0.5])

        # feet: a rising contact force = a foot landing
        for i, key in enumerate(('fl', 'fr')):
            fnow = float(ramp[key][-1]); rise = fnow - self.prev_f[i]
            if self.prev_f[i] < 80 and rise > 150 and self.t - self.last_step[i] > 0.25 * self.fs:
                self.last_step[i] = self.t
                self.pending.append([int(self.rng.integers(0, max(n // 4, 1))), 'step', min(rise / 1500.0, 1.0)])
            self.prev_f[i] = fnow
        click, thud, clank = self._events(n)
        self.t += n

        # relief valve: unstable shriek
        scr_amp = np.clip((Ps - P_RELIEF) / 10.0, 0, 1)
        if scr_amp.max() > 0:
            self.relief.set(*_bp((2900 + 180 * np.sin(self.ph[8])) / slow, 30)); self.ph[8] += 0.7
            scr = self.relief(self.rng.standard_normal(n)) * scr_amp * 0.9
        else: scr = np.zeros(n)

        dry = (pump + motor + rumble + pwm + hiss + scr) * on + (click * 0.35 + thud + clank * 0.6)
        dry = dry * ramp['gain']
        # stereo: pump / motor slightly left, valves and feet slightly right, plus the room
        st = np.vstack([dry + 0.15 * hiss * on, dry + 0.15 * pump * on])
        wet = self._reverb(dry)
        y = 0.86 * st + 0.22 * wet
        # bus compressor (slow envelope) + soft limiter
        lvl = np.sqrt(np.mean(y * y)) + 1e-9
        self.env += (lvl - self.env) * (0.3 if lvl > self.env else 0.05)
        g = 1.0 / (1.0 + max(self.env - 0.18, 0.0) * 3.0)
        y = np.tanh(y * g * 1.2) * 0.9
        return y.T.astype(np.float32)


class RobotSound:
    """live audio output; update() from the viewer loop, toggle() mutes"""
    def __init__(self):
        import sounddevice as sd                         # raises if PortAudio is missing
        self.syn = Synth(); self.muted = False; self.lock = threading.Lock()
        def cb(out, frames, tinfo, status):
            with self.lock: y = self.syn.block(frames)
            out[:] = 0 if self.muted else y
        self.stream = sd.OutputStream(samplerate=FS, channels=2, blocksize=512, dtype='float32', callback=cb, latency='low')
        self.stream.start()

    def update(self, **kw):
        with self.lock: self.syn.target.update({k: float(v) for k, v in kw.items()})

    def toggle(self): self.muted = not self.muted; return self.muted

    def close(self):
        try: self.stream.stop(); self.stream.close()
        except Exception: pass


def try_open():
    try: return RobotSound()
    except Exception as e:
        print(f'[sound] off ({e}).  enable: pip install sounddevice  &&  sudo apt install libportaudio2'); return None


def demo(path, seconds=12.0, synth=None):
    """offline render of a scripted run (spin-up, walking with steps and valve activity, relief, power off) -> WAV"""
    import wave
    syn = synth or Synth(); n = 512; out = []
    for b in range(int(seconds * FS / n)):
        t = b * n / FS; gait = (t * 1.0) % 1.0                      # one step per 0.5 s
        rpm = 3000 * min(t / 2.0, 1.0) if t < 10.5 else max(3000 * (1 - (t - 10.5) / 1.2), 0)
        Ps = 207 * min(t / 2.5, 1.0) if t < 10.5 else max(207 * (1 - (t - 10.5) / 0.8), 0)
        if 8.5 < t < 9.4: Ps = 228                                   # relief
        walking = 2.5 < t < 8.5
        Q = (6 + 18 * abs(np.sin(np.pi * gait * 2)) ** 3) if walking else 3
        fl = 900 if walking and gait < 0.5 else 0
        fr = 900 if walking and gait >= 0.5 else 0
        syn.target.update(rpm=rpm, Ps=Ps, Pel=4000 + 9000 * (Q / 24), Q=Q, fl=fl, fr=fr, on=1.0 if t < 10.5 else 0.0)
        y = syn.block(n)
        out.append(y if y.ndim == 2 else np.repeat(y[:, None], 2, 1))
    y = np.concatenate(out); y = y / max(np.abs(y).max(), 1e-6) * 0.89
    w = wave.open(path, 'wb'); w.setnchannels(2); w.setsampwidth(2); w.setframerate(FS)
    w.writeframes((y * 32767).astype('<i2').tobytes()); w.close()
    return y


if __name__ == '__main__':
    if '--demo' in sys.argv:
        p = sys.argv[sys.argv.index('--demo') + 1] if len(sys.argv) > sys.argv.index('--demo') + 1 else 'sound_demo.wav'
        demo(p); print('wrote', p)
