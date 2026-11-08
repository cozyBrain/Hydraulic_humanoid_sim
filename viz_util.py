"""Poses the moving linkage / cylinder parts (mocap bodies of the _viz model) from the joint angles,
using the same exact kinematics the hydraulic model uses."""
import json, os, numpy as np, mujoco
import kinematics as K
HERE = os.path.dirname(os.path.abspath(__file__))
U = K.U
JP = {k: np.array(v) for k, v in json.load(open(f'{HERE}/model/joint_points.json')).items()}
KN = K.KN

def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def rot_between(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)
def mat2quat(R):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.reshape(-1)); return q

class LinkagePoser:
    def __init__(self, m):
        self.m = m
        self.bid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in JP}
        self.mocap = {}
        for b in range(m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            if n and n.startswith('mc_'): self.mocap[n[3:]] = m.body_mocapid[b]
        self.jq = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j): m.jnt_qposadr[j] for j in range(m.njnt)}
        self.ph0 = KN['phi0']
        _, R10, R20, S0 = K.knee_pose2d(0.0)
        self.B2 = np.array([KN['yb'], KN['zb']]); self.Q2 = np.array([KN['yq'], KN['zq']])
        self.R10, self.R20, self.S0 = R10, R20, S0
    def _set(self, part, pos, R, d):
        i = self.mocap.get(part)
        if i is None or i < 0: return
        d.mocap_pos[i] = pos; d.mocap_quat[i] = mat2quat(R)
    def __call__(self, d, pump_angle=0.0):
        for s in 'LR':
            th = d.qpos[self.jq[f'knee_{s}']]
            Tb = self.bid[f'thigh_{s}']; Xt, Rt = d.xpos[Tb], d.xmat[Tb].reshape(3, 3)
            Kt = JP[f'shin_{s}'] - JP[f'thigh_{s}']                      # knee pin in thigh frame
            to3 = lambda p2: Kt + np.array([p2[1] * U, 0, p2[0] * U])    # (y3, z3) design units -> thigh-frame metres
            ang = lambda v: np.arctan2(v[1], v[0])
            phi, R1, R2, S = K.knee_pose2d(th)
            W = lambda p: Xt + Rt @ p
            self._set(f'rocker_{s}', W(to3(self.Q2)), Rt @ Ry(phi - self.ph0), d)
            self._set(f'coupler_{s}', W(to3(R2)), Rt @ Ry(ang(S - R2) - ang(self.S0 - self.R20)), d)
            Rb = Rt @ Ry(ang(R1 - self.B2) - ang(self.R10 - self.B2)); L0 = np.linalg.norm(self.R10 - self.B2) * U; L = np.linalg.norm(R1 - self.B2) * U
            un = (R1 - self.B2) / np.linalg.norm(R1 - self.B2); unw = Rt @ np.array([un[1], 0, un[0]])
            self._set(f'quadbarrel_{s}', W(to3(self.B2)), Rb, d)
            self._set(f'quadrod_{s}', W(to3(self.B2)) + unw * (L - L0), Rb, d)
            # Achilles (shin frame)
            Sb = self.bid[f'shin_{s}']; Xs, Rs = d.xpos[Sb], d.xmat[Sb].reshape(3, 3)
            qp, qr = d.qpos[self.jq[f'ankle_pitch_{s}']], d.qpos[self.jq[f'ankle_roll_{s}']]
            A, T, H = K.ach_points(s); O = JP[f'shin_{s}']
            for i in range(2):
                Hq = Ry(qp) @ Rx(qr) @ (H[i] - A) + A
                d0, d1 = H[i] - T[i], Hq - T[i]; Rr = rot_between(d0, d1)
                self._set(f'achbarrel_{s}{i}', Xs + Rs @ (T[i] - O), Rs @ Rr, d)
                self._set(f'achrod_{s}{i}', Xs + Rs @ (T[i] - O + d1 / np.linalg.norm(d1) * (np.linalg.norm(d1) - np.linalg.norm(d0))), Rs @ Rr, d)
        # triceps cylinders (upper-arm frame)
        for s in 'LR':
            if f'tribarrel_{s}' not in self.mocap: continue
            Ub = self.bid[f'uarm_{s}']; Xu, Ru = d.xpos[Ub], d.xmat[Ub].reshape(3, 3); O = JP[f'uarm_{s}']
            pin = JP[f'farm_{s}'] - O; sg = 1 if s == 'L' else -1
            B = np.array([-0.55 * U, sg * 2.98 * U, 4.55 * U + K.ZG]) - O
            to3 = lambda p2: pin + np.array([p2[1] * U, 0, p2[0] * U])
            H0 = to3(K.EL_H0); H = to3(K.elbow_pose2d(d.qpos[self.jq[f'elbow_{s}']]))
            d0, d1 = H0 - B, H - B; Rr = rot_between(d0, d1)
            self._set(f'tribarrel_{s}', Xu + Ru @ B, Ru @ Rr, d)
            self._set(f'trirod_{s}', Xu + Ru @ (B + d1 / np.linalg.norm(d1) * (np.linalg.norm(d1) - np.linalg.norm(d0))), Ru @ Rr, d)
        # pump: pinion + shaft + coupling turn with the motor, the eccentric ring gear at 17/23 of that speed
        Tb = self.bid['torso']; Xt, Rt = d.xpos[Tb], d.xmat[Tb].reshape(3, 3); O = JP['torso']
        for part, ratio, org in (('pumpshaft', 1.0, (1.15, 4.0)), ('pumpring', 17 / 23, (1.15, 4.0 - 0.0967742 * 0.62))):
            if part in self.mocap:
                p = np.array([0.634 * U, org[0] * U, org[1] * U + K.ZG]) - O
                self._set(part, Xt + Rt @ p, Rt @ Rx(pump_angle * ratio), d)
        # waist cylinders (carrier frame)
        Cb = self.bid['carrier']; Xc, Rc = d.xpos[Cb], d.xmat[Cb].reshape(3, 3); O = JP['carrier']
        qp, qr = d.qpos[self.jq['waist_pitch']], d.qpos[self.jq['waist_roll']]
        for i, s in enumerate('LR'):
            Wb, E0 = K.WB[i], K.WE[i]; E = Ry(qp) @ Rx(qr) @ (E0 - K.U_C) + K.U_C
            d0, d1 = E0 - Wb, E - Wb; Rr = rot_between(d0, d1)
            self._set(f'waistbarrel_{s}', Xc + Rc @ (Wb - O), Rc @ Rr, d)
            self._set(f'waistrod_{s}', Xc + Rc @ (Wb - O + d1 / np.linalg.norm(d1) * (np.linalg.norm(d1) - np.linalg.norm(d0))), Rc @ Rr, d)
