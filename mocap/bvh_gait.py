"""CMU BVH -> per-gait-cycle sagittal joint angles (hip/knee/ankle/shoulder/elbow), phase-normalised and averaged.
usage: python3 -I bvh_gait.py OUT.npz file1.bvh file2.bvh ..."""
import sys, numpy as np

def parse(path):
    lines = open(path).read().split('\n'); i = 0
    joints = []; stack = []; chans = []
    while not lines[i].strip().startswith('MOTION'):
        tok = lines[i].split()
        if tok and tok[0] in ('ROOT', 'JOINT'):
            joints.append(dict(name=tok[1], parent=stack[-1] if stack else -1, offset=None, ch=[])); cur = len(joints) - 1
        elif tok and tok[0] == 'End':
            joints.append(dict(name=joints[stack[-1]]['name'] + '_end', parent=stack[-1], offset=None, ch=[])); cur = len(joints) - 1
        elif tok and tok[0] == '{': stack.append(cur)
        elif tok and tok[0] == '}': stack.pop()
        elif tok and tok[0] == 'OFFSET': joints[cur]['offset'] = np.array([float(x) for x in tok[1:4]])
        elif tok and tok[0] == 'CHANNELS':
            joints[cur]['ch'] = tok[2:]; joints[cur]['c0'] = len(chans); chans += [(cur, c) for c in tok[2:]]
        i += 1
    nf = int(lines[i + 1].split()[1]); dt = float(lines[i + 2].split()[2])
    M = np.array([[float(x) for x in l.split()] for l in lines[i + 3:i + 3 + nf] if l.strip()])
    return joints, M, dt

def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    return {'X': np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), 'Y': np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            'Z': np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]

def fk(joints, M):
    n = len(joints); F = len(M); P = np.zeros((F, n, 3)); R = np.zeros((F, n, 3, 3))
    for f in range(F):
        for j, J in enumerate(joints):
            Rl = np.eye(3); t = J['offset'].copy()
            for k, c in enumerate(J['ch']):
                v = M[f, J['c0'] + k]
                if c.endswith('position'): t = t + 0  # root translation handled below
                else: Rl = Rl @ rot(c[0], np.radians(v))
            if J['parent'] < 0:
                pos = np.array([M[f, J['c0'] + k] for k, c in enumerate(J['ch']) if c.endswith('position')])
                P[f, j] = pos; R[f, j] = Rl
            else:
                p = J['parent']; P[f, j] = P[f, p] + R[f, p] @ J['offset']; R[f, j] = R[f, p] @ Rl
    return P, R

def ang(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1), -1, 1)))

def angles(path):
    joints, M, dt = parse(path); P, R = fk(joints, M); ix = {J['name']: k for k, J in enumerate(joints)}
    Rp = R[:, ix['Hips']]; up = Rp[:, :, 1]; fwd = Rp[:, :, 2]; left = Rp[:, :, 0]
    out = {}
    for s, side in (('L', 'Left'), ('R', 'Right')):
        hip, knee, ank, toe = (P[:, ix[n]] for n in (f'{side}UpLeg', f'{side}Leg', f'{side}Foot', f'{side}ToeBase'))
        th, sh, ft = knee - hip, ank - knee, toe - ank
        out[f'hip_{s}'] = np.degrees(np.arctan2(np.sum(th * fwd, 1), -np.sum(th * up, 1)))
        out[f'knee_{s}'] = ang(th, sh)
        J = {k: joints[ix[n]]['offset'] for k, n in (('sh', f'{side}Foot'), ('ft', f'{side}ToeBase'))}
        neutral = ang(-J['sh'][None], J['ft'][None])[0]
        out[f'dorsi_{s}'] = neutral - ang(-sh, ft)
        shd, elb, wr = (P[:, ix[n]] for n in (f'{side}Arm', f'{side}ForeArm', f'{side}Hand'))
        ua, fa = elb - shd, wr - elb
        out[f'shoulder_{s}'] = np.degrees(np.arctan2(np.sum(ua * fwd, 1), -np.sum(ua * up, 1)))
        Rt = R[:, ix['Spine1']]; tup, tfwd = Rt[:, :, 1], Rt[:, :, 2]          # arm swing measured in the THORAX frame
        out[f'shoulderT_{s}'] = np.degrees(np.arctan2(np.sum(ua * tfwd, 1), -np.sum(ua * tup, 1)))
        out[f'elbow_{s}'] = ang(ua, fa)
        out[f'footfwd_{s}'] = np.sum((ank - P[:, ix['Hips']]) * fwd, 1)
        sgn = 1.0 if s == 'L' else -1.0
        out[f'hipab_{s}'] = np.degrees(np.arctan2(sgn * np.sum(th * left, 1), -np.sum(th * up, 1)))   # abduction +
    # thorax (Spine1) orientation relative to the pelvis: yaw (twist), roll, pitch
    Rrel = np.einsum('fji,fjk->fik', Rp, R[:, ix['Spine1']])
    out['th_yaw'] = np.degrees(np.arctan2(Rrel[:, 0, 2], Rrel[:, 2, 2]))       # rotation about the up axis
    out['th_roll'] = np.degrees(np.arctan2(Rrel[:, 0, 1], Rrel[:, 1, 1]))      # sideways bend (about forward axis)
    out['th_pitch'] = np.degrees(np.arctan2(-Rrel[:, 2, 1], Rrel[:, 1, 1]))    # forward bend
    root = P[:, ix['Hips']]; hv = root[-1] - root[0]; hv[1] = 0
    # pelvis motion in the heading frame (straight walking trials): yaw/roll/pitch + lateral sway + vertical bob
    Xh = hv / np.linalg.norm(hv); Zu = np.array([0.0, 1.0, 0.0]); Yh = np.cross(Zu, Xh)          # Yh = left of heading
    out['pel_yaw'] = np.degrees(np.arctan2(fwd @ Yh, fwd @ Xh)); out['pel_yaw'] -= out['pel_yaw'].mean()
    out['pel_roll'] = np.degrees(np.arcsin(np.clip(left @ Zu, -1, 1)))          # left side up = +
    out['pel_pitch'] = np.degrees(np.arcsin(np.clip(fwd @ Zu, -1, 1)))          # nose up (posterior tilt) = +
    t_ = np.arange(len(root)); along = (root - root[0]) @ Xh
    lat = (root - root[0]) @ Yh; lat = lat - np.polyval(np.polyfit(t_, lat, 1), t_)
    leg_u = np.linalg.norm(joints[ix['LeftLeg']]['offset']) + np.linalg.norm(joints[ix['LeftFoot']]['offset'])
    out['pel_lat'] = lat / leg_u; out['pel_up'] = (root[:, 1] - root[:, 1].mean()) / leg_u          # in leg lengths
    leg = np.linalg.norm(joints[ix['LeftLeg']]['offset']) + np.linalg.norm(joints[ix['LeftFoot']]['offset'])
    out['speed_leglen'] = np.linalg.norm(hv) / (len(root) * dt) / leg      # speed in leg lengths per second (unit-free)
    return out, dt

def cycles(out, dt, n=101):
    ff = out['footfwd_L']; k = 12
    pk = [i for i in range(k, len(ff) - k) if ff[i] == ff[i - k:i + k + 1].max()]   # left foot most forward ~ heel strike
    curves, T = [], []
    keys = [k_ for k_ in out if not k_.startswith('footfwd') and k_ != 'speed_leglen']
    for a, b in zip(pk[:-1], pk[1:]):
        t = np.linspace(a, b, n); T.append((b - a) * dt)
        curves.append(np.stack([np.interp(t, np.arange(len(out[k_])), out[k_]) for k_ in keys], 1))
    return keys, curves, T

if __name__ == '__main__':
    OUT = sys.argv[1]; allc, allT, spd = [], [], []
    for f in sys.argv[2:]:
        out, dt = angles(f); keys, C, T = cycles(out, dt)
        allc += C; allT += T; spd.append(out['speed_leglen'])
        print(f.split('/')[-1], f'cycles {len(C)}  stride {np.mean(T) if T else 0:.2f}s  speed {out["speed_leglen"]:.2f} leglen/s')
    C = np.array(allc); mean = C.mean(0); std = C.std(0)
    np.savez(OUT, keys=np.array(keys), mean=mean, std=std, stride_T=np.mean(allT), speed_leglen=np.mean(spd), n=len(C))
    print('total cycles', len(C), 'mean stride', round(float(np.mean(allT)), 3), 's  speed', round(float(np.mean(spd)), 3), 'leglen/s')
    for j, k_ in enumerate(keys): print(f'  {k_:12s} min {mean[:, j].min():6.1f}  max {mean[:, j].max():6.1f}  mean std {std[:, j].mean():4.1f}')
