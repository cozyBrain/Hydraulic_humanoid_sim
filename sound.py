"""Real-time robot sound for viewer.py (demo / testing only, the trainer never uses it).

Everything is synthesised from the simulation state, so pitch and loudness follow the physics:
  - gear pump whine   : tooth-mesh frequency = pump rpm / 60 x 17 teeth (pinion of the CAD pump) + harmonics,
                        loudness ~ supply pressure (pressure ripple is what you hear on a real gear pump)
  - EMRAX motor       : electrical frequency = rpm / 60 x pole pairs, 6th-harmonic torque ripple, loudness ~ torque
  - inverter whine    : PWM switching tone (8 kHz) modulated by the motor current
  - servo valve hiss  : band-passed noise, loudness ~ valve flow x sqrt(pressure drop)
  - relief valve      : screech when the supply pressure reaches the relief setting
Slow motion (G key) slows the sound with the simulation: all frequencies are divided by the slow-down factor.

needs:  pip install sounddevice    and on Linux Mint:  sudo apt install libportaudio2
"""
import threading, numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

FS = 44100
PUMP_TEETH = 17          # pinion of the internal gear pump (robot24.html Z1, viz_util ratio 17/23)
POLE_PAIRS = 10          # EMRAX axial-flux rotor (assumed 10 pole pairs)
F_PWM = 8000.0           # inverter switching frequency (emDrive: 4..16 kHz configurable)
P_RELIEF = 225.0         # bar, screech starts just below the 230 bar relief


class Synth:
    """block synthesiser: call block(n) to get n samples; set .target dict from any thread"""
    def __init__(self, fs=FS):
        self.fs = fs
        self.target = dict(rpm=0.0, Ps=0.0, Pel=0.0, Q=0.0, slow=1.0, on=1.0, gain=1.0)
        self.cur = dict(self.target)
        self.ph = np.zeros(12)                         # oscillator phases (continuous across blocks)
        self.rng = np.random.default_rng(0)
        self.b_hiss, self.a_hiss = butter(2, [1500 / (fs / 2), 7000 / (fs / 2)], 'band')
        self.z_hiss = lfilter_zi(self.b_hiss, self.a_hiss) * 0
        self.b_low, self.a_low = butter(2, 400 / (fs / 2), 'low')
        self.z_low = lfilter_zi(self.b_low, self.a_low) * 0

    def _osc(self, k, f, n):
        """phase-continuous sine of (time-varying) frequency array f"""
        ph = self.ph[k] + 2 * np.pi * np.cumsum(f) / self.fs
        self.ph[k] = ph[-1] % (2 * np.pi)
        return np.sin(ph)

    def block(self, n):
        t, c = self.target, self.cur
        a = np.linspace(0, 1, n)                      # smooth parameter ramps inside the block (no zipper noise)
        ramp = {}
        for k in c:
            ramp[k] = c[k] + (t[k] - c[k]) * a
            c[k] = c[k] + (t[k] - c[k])
        slow = np.maximum(ramp['slow'], 1.0); on = ramp['on']
        rps = ramp['rpm'] / 60.0 / slow
        Ps = np.clip(ramp['Ps'], 0, 400); load = np.clip(ramp['Pel'] / 15000.0, 0, 1.5)
        # gear pump: mesh tone + 2 harmonics, pressure ripple makes it louder under load
        fm = rps * PUMP_TEETH
        amp_p = (0.03 + 0.20 * Ps / 207.0) * np.clip(rps * slow / 10.0, 0, 1)
        # rich harmonic series: at idle the mesh tone is only ~40 Hz, which laptop speakers cannot play -> the
        # upper harmonics (gear-tooth impacts) carry the sound, like on a real gear pump
        pump = amp_p * (self._osc(0, fm, n) + 0.6 * self._osc(1, 2 * fm, n) + 0.45 * self._osc(2, 3 * fm, n)
                        + 0.35 * self._osc(7, 5 * fm, n) + 0.25 * np.sign(self._osc(8, 4 * fm, n)) * 0.6)
        # EMRAX: electrical frequency + 6th harmonic torque ripple
        fe = rps * POLE_PAIRS
        motor = (0.02 + 0.10 * load) * (self._osc(3, fe, n) + 0.4 * self._osc(4, 6 * fe, n)) * np.clip(rps * slow / 5, 0, 1)
        # inverter PWM whine (not slowed: it is the electronics, but it stops when power is off)
        pwm = 0.012 * np.clip(load * 3, 0, 1) * self._osc(5, np.full(n, F_PWM), n) * (slow < 1.5)
        # servo valve hiss: flow through the metering edges (high pressure drop -> noise)
        hiss_amp = np.clip(0.06 * np.sqrt(np.maximum(ramp['Q'], 0) / 20.0) * np.sqrt(Ps / 207.0), 0, 0.25)
        noise = self.rng.standard_normal(n)
        if slow[0] > 1.5:                               # slowed: move the hiss down in frequency
            nz, self.z_low = lfilter(self.b_low, self.a_low, noise, zi=self.z_low)
        else:
            nz, self.z_hiss = lfilter(self.b_hiss, self.a_hiss, noise, zi=self.z_hiss)
        hiss = hiss_amp * nz * 3.0
        # relief valve screech
        scr = np.clip((Ps - P_RELIEF) / 10.0, 0, 1) * 0.15 * self._osc(6, np.full(n, 2900.0) / slow, n)
        # cooler fan + mechanical hum: constant while hydraulic power is on (so you always hear that it is alive)
        fan = 0.025 * nz * 0.5 + 0.02 * self._osc(9, np.full(n, 150.0) / slow, n)
        y = (pump + motor + pwm + hiss + scr + fan) * on * ramp['gain']
        return np.tanh(y * 1.5).astype(np.float32)   # soft clip


class RobotSound:
    """live audio output; update() from the viewer loop, toggle() mutes"""
    def __init__(self):
        import sounddevice as sd                         # raises if PortAudio is missing
        self.syn = Synth(); self.muted = False; self.lock = threading.Lock()
        def cb(out, frames, tinfo, status):
            with self.lock: y = self.syn.block(frames)
            out[:, 0] = 0 if self.muted else y
        self.stream = sd.OutputStream(samplerate=FS, channels=1, blocksize=512, dtype='float32', callback=cb, latency='low')
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
