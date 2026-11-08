"""Exact actuator kinematics (lengths, moment arms) from the CAD geometry.  MuJoCo frame, metres, radians.
knee  : 4-bar (cylinder -> bell-crank rocker -> coupler -> shin), tabulated from knee_de3 synthesis
ankle : two Achilles cylinders on a cross U-joint (pitch, roll)
waist : two back cylinders on the waist cross U-joint (pitch, roll)"""
import json, os, numpy as np
import jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__))
from legs import LY, ZG, U
LX = 0.85
def P(x3, y3, z3): return np.array([z3 * U, x3 * U, y3 * U + ZG])
KN = json.load(open(f'{HERE}/model/knee_solution.json')) if os.path.exists(f'{HERE}/model/knee_solution.json') else json.load(open(f'{HERE}/../knee_sol3_21.json'))

# ------------------------------------------------------------------ knee 4-bar (design units, (y3,z3) about the knee pin)
def _rot(v, a): return np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
def knee_solve(thetas):
    Q, S0, B = np.array([KN['yq'], KN['zq']]), np.array([KN['ys'], KN['zs']]), np.array([KN['yb'], KN['zb']])
    phi = KN['phi0']; out = []
    grid = np.unique(np.concatenate([np.linspace(0, t, max(2, int(abs(t) / np.radians(0.25)) + 1)) for t in [thetas.min(), thetas.max()]]))
    # march from 0 both directions for branch continuity
    res = {}
    for direction in (1, -1):
        phi = KN['phi0']; ts = np.sort(grid[grid >= 0]) if direction > 0 else np.sort(grid[grid <= 0])[::-1]
        for t in ts:
            Sp = _rot(S0, t); d = np.linalg.norm(Sp - Q)
            a = (KN['r2'] ** 2 - KN['Lc'] ** 2 + d * d) / (2 * d); h = np.sqrt(max(KN['r2'] ** 2 - a * a, 0))
            u = (Sp - Q) / d; mid = Q + a * u; pp = np.array([-u[1], u[0]]); best = None
            for sg in (1, -1):
                c = mid + sg * h * pp; dp = np.angle(np.exp(1j * (np.arctan2(c[1] - Q[1], c[0] - Q[0]) - phi)))
                if best is None or abs(dp) < abs(best[1]): best = (c, dp)
            phi += best[1]; R1 = Q + KN['r1'] * np.array([np.cos(phi + KN['beta']), np.sin(phi + KN['beta'])])
            res[round(float(t), 9)] = (np.linalg.norm(R1 - B), phi, R1, best[0], Sp)
    ks = sorted(res); return np.array(ks), res
KTH, _KRES = knee_solve(np.radians(np.array([-3.0, 143.0])))
KL = np.array([_KRES[k][0] for k in KTH]) * U                      # cylinder length (m)
KARM = np.gradient(KL, KTH)                                          # dL/dtheta (m/rad)  (>0: extends with flexion)
KPHI = np.array([_KRES[k][1] for k in KTH])
def knee_pose2d(theta):   # for the viewer: rocker angle, R1, R2, S (design units, knee-pin frame)
    i = int(np.clip(np.searchsorted(KTH, theta), 1, len(KTH) - 1)); r = _KRES[KTH[i]]
    return r[1], r[2], r[3], r[4]
_KTH_j, _KL_j, _KARM_j = jnp.array(KTH), jnp.array(KL), jnp.array(KARM)
def knee_len_arm(theta):
    return jnp.interp(theta, _KTH_j, _KL_j), jnp.interp(theta, _KTH_j, _KARM_j)
KNEE_LMIN, KNEE_LMAX = float(KL[(KTH >= 0) & (KTH <= np.radians(140))].min()), float(KL[(KTH >= 0) & (KTH <= np.radians(140))].max())

# ------------------------------------------------------------------ rotations (MuJoCo hinge convention)
def Ry(a): c, s = jnp.cos(a), jnp.sin(a); return jnp.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rx(a): c, s = jnp.cos(a), jnp.sin(a); return jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])

# ------------------------------------------------------------------ Achilles (per side: 2 cylinders), shin frame
def ach_points(side):
    sg = 1 if side == 'L' else -1; x = sg * LX; A = P(x, LY(-4.5), 0.05)
    T = np.stack([P(x + xo, LY(-2.96), -0.3) for xo in (-0.11, 0.11)]); H = np.stack([P(x + xo, LY(-4.56), -0.42) for xo in (-0.11, 0.11)])
    return A, T, H
_ACH = {s: tuple(jnp.array(v) for v in ach_points(s)) for s in 'LR'}
def ach_lengths(qp, qr, side):
    A, T, H = _ACH[side]; R = Ry(qp) @ Rx(qr)
    Hw = (R @ (H - A).T).T + A
    return jnp.linalg.norm(Hw - T, axis=1)
def ach_len_jac(qp, qr, side):
    L = ach_lengths(qp, qr, side); Jm = jax.jacfwd(lambda q: ach_lengths(q[0], q[1], side))(jnp.array([qp, qr]))
    return L, Jm                                                     # (2,), (2 cyl, 2 dof)
ACH_L0 = float(np.linalg.norm(ach_points('L')[2][0] - ach_points('L')[1][0]))

# ------------------------------------------------------------------ waist (2 back cylinders), carrier frame
U_C = P(0, 1.1, -0.2)
WB = np.stack([P(sx * 0.62, 0.84, -0.62) for sx in (1, -1)]); WE = np.stack([P(sx * 0.82, 2.12, -0.72) for sx in (1, -1)])
_WB, _WE, _UC = jnp.array(WB), jnp.array(WE), jnp.array(U_C)
def waist_lengths(qp, qr):
    R = Ry(qp) @ Rx(qr); E = (R @ (_WE - _UC).T).T + _UC
    return jnp.linalg.norm(E - _WB, axis=1)
def waist_len_jac(qp, qr):
    L = waist_lengths(qp, qr); Jm = jax.jacfwd(lambda q: waist_lengths(q[0], q[1]))(jnp.array([qp, qr]))
    return L, Jm
WAIST_L0 = float(np.linalg.norm(WE[0] - WB[0]))

if __name__ == '__main__':
    print('knee  L(0)=%.1f mm  L(140)=%.1f mm  arm %.1f..%.1f mm/rad' % (KL[np.argmin(abs(KTH))] * 1e3, np.interp(np.radians(140), KTH, KL) * 1e3,
          KARM[(KTH >= 0) & (KTH <= 2.44)].min() * 1e3, KARM[(KTH >= 0) & (KTH <= 2.44)].max() * 1e3))
    for qp in (-0.5, 0, 0.5):
        L, Jm = ach_len_jac(qp, 0.0, 'L'); print('ach qp=%+.1f L=%s mm  dL/dq=%s mm/rad' % (qp, np.round(np.array(L) * 1e3, 1), np.round(np.array(Jm) * 1e3, 1).tolist()))
    for qp in (-0.4, 0, 0.4):
        L, Jm = waist_len_jac(qp, 0.0); print('waist qp=%+.1f L=%s mm  dL/dq=%s' % (qp, np.round(np.array(L) * 1e3, 1), np.round(np.array(Jm) * 1e3, 1).tolist()))

# ------------------------------------------------------------------ elbow: triceps cylinder on the olecranon horn (upper-arm frame)
# design units about the elbow pin, (y3, z3); flexion theta>0 swings the forearm forward
EL_H0 = np.array([0.17, -0.36]); EL_B = np.array([1.40, -0.35])
def _elbow_L(th):
    c, s = jnp.cos(-th), jnp.sin(-th)
    H = jnp.array([EL_H0[0] * c - EL_H0[1] * s, EL_H0[0] * s + EL_H0[1] * c])
    return jnp.linalg.norm(H - jnp.array(EL_B)) * U
def elbow_len_arm(th):
    return _elbow_L(th), jax.grad(_elbow_L)(th)
ELBOW_LMIN, ELBOW_LMAX = float(_elbow_L(0.0)), float(_elbow_L(np.radians(110)))
def elbow_pose2d(th):
    c, s = np.cos(-th), np.sin(-th); return np.array([EL_H0[0] * c - EL_H0[1] * s, EL_H0[0] * s + EL_H0[1] * c])
