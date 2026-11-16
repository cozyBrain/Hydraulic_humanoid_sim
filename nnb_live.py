"""Send a running simulation's policy inputs to NNB, so the network can be watched while the robot moves.

NNB listens on UDP 127.0.0.1:7401 while a space is open (R_NetBlocks). Each packet carries the observation the
policy just received (raw, as walk_env.obs_core returns it; NNB normalises it with the block's own norm_mean/std)
and, optionally, the action the simulation applied, which NNB compares with its own output.

Only the standard library is used (numpy arrays are accepted). Sending never blocks and never fails the caller:
if NNB is not running, packets are simply dropped.

    from nnb_live import NNBLink
    link = NNBLink()                       # 127.0.0.1:7401
    link.send(obs, act, t=d.time, qpos=d.qpos, pump_w=w)   # obs: 128 floats, act: 29 floats (or None); qpos, pump_w optional

Packet (little-endian): "NNBL" u8 version=1, u8 flags, u16 n_obs, u16 n_act, s16 block (0 = first block whose
input size is n_obs), f64 sim time, f32 obs[n_obs], f32 act[n_act];
flags bit 0: then u16 n_qpos, f32 qpos[n_qpos] (the robot's full MuJoCo qpos: NNB poses its robot preview with it);
flags bit 1: then f32 pump shaft speed [rad/s] (NNB turns the preview's pump in proportion);
flags bit 2: the action came from the network as edited in NNB (see below).

Edited brain in the simulation: NNB answers on the same socket with the edits made to the block (switched-off neurons,
fixed values), as "NNBE" u8 version=1 + UTF-8 JSON {"on": bool, "off": [[layer, index], ...], "fixed": [[layer, index,
activation], ...]} (layer 0 = input, 1.. = dense layers), whenever they change and once a second. The simulation then runs
the policy itself with those edits applied:
    link.use_policy(params)                 # once: the Brax params (normaliser, policy, value) the viewer loaded
    a = link.edited_action(obs) or a        # each policy step: None while NNB has no edits switched on
"""
import socket
import struct

HEADER = struct.Struct('<4sBBHHhd')   # 20 bytes
DEFAULT_PORT = 7401


class NNBLink:
    def __init__(self, host='127.0.0.1', port=DEFAULT_PORT, block=0):
        self.addr = (host, port)
        self.block = block
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sent = 0
        self.edits = {}          # newest edits from NNB ({} = none)
        self.arrays = None       # (mean, std, [(W, b, activation)]) of the policy, for edited_action
        self.edited = False      # the last action sent was computed with edits

    def send(self, obs, act=None, t=0.0, qpos=None, pump_w=None):
        obs = [float(x) for x in obs]
        act = [] if act is None else [float(x) for x in act]
        flags = (0 if qpos is None else 1) | (0 if pump_w is None else 2) | (4 if self.edited else 0)
        pkt = HEADER.pack(b'NNBL', 1, flags, len(obs), len(act), self.block, float(t))
        pkt += struct.pack('<%df' % (len(obs) + len(act)), *obs, *act)
        if qpos is not None:
            q = [float(x) for x in qpos]
            pkt += struct.pack('<H%df' % len(q), len(q), *q)
        if pump_w is not None:
            pkt += struct.pack('<f', float(pump_w))
        try:
            self.sock.sendto(pkt, self.addr)
            self.sent += 1
        except OSError:
            pass   # NNB not running / buffer full: the simulation must never stall because of the viewer

    # ---- edited brain ---------------------------------------------------------------------------------------------

    def poll(self):
        """Read NNB's answers (non-blocking): keeps the newest edits."""
        import json
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except (BlockingIOError, OSError):
                return
            if data[:4] == b'NNBE' and data[4] == 1:
                try:
                    self.edits = json.loads(data[5:].decode('utf-8'))
                except ValueError:
                    pass

    def use_policy(self, params):
        """Brax PPO params as pickled by training: (normaliser state, {'params': {hidden_i: {kernel, bias}}}, ...)."""
        import numpy as np
        norm, pol = params[0], params[1]['params']
        get = lambda o, k: o[k] if isinstance(o, dict) else getattr(o, k)
        mean = np.asarray(get(norm, 'mean')['state'], np.float64); std = np.asarray(get(norm, 'std')['state'], np.float64)
        n = len(pol)
        layers = [(np.asarray(pol[f'hidden_{i}']['kernel'], np.float64), np.asarray(pol[f'hidden_{i}']['bias'], np.float64),
                   'swish' if i < n - 1 else 'linear') for i in range(n)]
        self.arrays = (mean, std, layers)

    def edited_action(self, obs):
        """The policy's deterministic action (tanh of the first half of the outputs) with NNB's edits, or None."""
        self.poll()
        e = self.edits
        if self.arrays is None or not e.get('on') or not (e.get('off') or e.get('fixed')):
            self.edited = False
            return None
        import numpy as np
        mean, std, layers = self.arrays
        off = {}; fixed = {}
        for l, i in e.get('off', []): off.setdefault(int(l), []).append(int(i))
        for l, i, v in e.get('fixed', []): fixed.setdefault(int(l), []).append((int(i), float(v)))
        h = (np.asarray(obs, np.float64) - mean) / std
        h[off.get(0, [])] = 0.0
        for k, (W, b, act) in enumerate(layers, start=1):
            z = h @ W + b
            h = z / (1.0 + np.exp(-z)) if act == 'swish' else z
            for i, v in fixed.get(k, []): h[i] = v
            h[off.get(k, [])] = 0.0
        self.edited = True
        na = len(h) // 2
        return np.clip(np.tanh(h[:na]), -1.0, 1.0)

    def close(self):
        self.sock.close()
