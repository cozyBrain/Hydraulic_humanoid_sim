"""Send a running simulation's policy inputs to NNB, so the network can be watched while the robot moves.

NNB listens on UDP 127.0.0.1:7401 while a space is open (R_NetBlocks). Each packet carries the observation the
policy just received (raw, as walk_env.obs_core returns it; NNB normalises it with the block's own norm_mean/std)
and, optionally, the action the simulation applied, which NNB compares with its own output.

Only the standard library is used (numpy arrays are accepted). Sending never blocks and never fails the caller:
if NNB is not running, packets are simply dropped.

    from nnb_live import NNBLink
    link = NNBLink()                       # 127.0.0.1:7401
    link.send(obs, act, t=d.time)          # obs: 128 floats, act: 29 floats (or None)

Packet (little-endian): "NNBL" u8 version=1, u8 flags=0, u16 n_obs, u16 n_act, s16 block (0 = first block whose
input size is n_obs), f64 sim time, f32 obs[n_obs], f32 act[n_act].
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

    def send(self, obs, act=None, t=0.0):
        obs = [float(x) for x in obs]
        act = [] if act is None else [float(x) for x in act]
        pkt = HEADER.pack(b'NNBL', 1, 0, len(obs), len(act), self.block, float(t))
        pkt += struct.pack('<%df' % (len(obs) + len(act)), *obs, *act)
        try:
            self.sock.sendto(pkt, self.addr)
            self.sent += 1
        except OSError:
            pass   # NNB not running / buffer full: the simulation must never stall because of the viewer

    def close(self):
        self.sock.close()
