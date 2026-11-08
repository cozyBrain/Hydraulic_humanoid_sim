"""mocap gait curves (bvh_gait.py) -> robot reference sim/gait_ref.npz
Joints: hip pitch/roll/yaw, knee, ankle pitch/roll, shoulder pitch, elbow (both sides) + waist yaw/roll.
Root:   pelvis roll / pitch / yaw (rad, oscillation about the heading), lateral sway / vertical bob (m).
The gait is symmetrised (left = right shifted by half a cycle) and Froude-scaled to the robot leg.
usage: python3 make_ref.py walk.npz ../sim/gait_ref.npz [human leg length m]"""
import sys, numpy as np
sys.path.insert(0, '/home/claude/tripo/sim')
import mujoco, sim_core as sc, hydraulics as hy

src, out = sys.argv[1], sys.argv[2]; L_h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
# ---- style: less hip sway, more upper-body drive (thorax counter-rotation, bigger arm swing)
PELVIS, THORAX, ARM = 0.6, 1.3, 2.0
L_r = 0.734                                          # robot hip-ankle length (v10 legs)
z = np.load(src); keys = list(z['keys']); C = z['mean']; n = len(C); H = (n - 1) // 2
g = lambda k: C[:, keys.index(k)]
half = lambda a: np.roll(a, -H)                      # value half a cycle later
sym = lambda l, r: 0.5 * (l + np.roll(r, H))         # left curve, symmetrised with the right one
anti = lambda a: 0.5 * (a - half(a))                 # signals that flip sign every step (roll, yaw, sway)
even = lambda a: 0.5 * (a + half(a))                 # signals that repeat every step (bob, pitch)
dm = lambda a: a - a.mean()

hip, knee, dor = sym(g('hip_L'), g('hip_R')), sym(g('knee_L'), g('knee_R')), sym(g('dorsi_L'), g('dorsi_R'))
shd = ARM * dm(sym(g('shoulderT_L'), g('shoulderT_R')))               # arm swing in the thorax frame
el0 = sym(g('elbow_L'), g('elbow_R')); elb = el0.mean() + 1.5 * dm(el0)
abd = PELVIS * dm(sym(g('hipab_L'), g('hipab_R')))   # left-leg abduction (+), oscillation only
p_roll, p_yaw = PELVIS * anti(dm(g('pel_roll'))), PELVIS * anti(dm(g('pel_yaw')))
p_pitch, p_up = dm(even(g('pel_pitch'))), dm(even(g('pel_up'))) * L_r
p_lat = PELVIS * anti(dm(g('pel_lat'))) * L_r
t_yaw = THORAX * anti(dm(g('th_yaw')))               # thorax twist relative to the pelvis (counter-rotation)
t_roll = -THORAX * anti(dm(g('th_roll')))            # thorax side-bend relative to the pelvis (robot sign)

R = lambda deg: np.radians(deg)
cols = {}
for s, sh in (('L', 0), ('R', H)):
    rl = lambda a: np.roll(a, -sh)                                   # right leg = left leg half a cycle later
    hr = rl(abd) if s == 'L' else -rl(abd)                           # robot hip_roll +: foot to +y (abduct L / adduct R)
    pr = p_roll                                                      # pelvis roll is a whole-body signal (not shifted)
    cols[f'hip_pitch_{s}'] = R(rl(hip)); cols[f'knee_{s}'] = R(rl(knee)); cols[f'ankle_pitch_{s}'] = R(-rl(dor))
    cols[f'hip_roll_{s}'] = R(hr); cols[f'ankle_roll_{s}'] = R(-(pr + hr))      # keep the sole level
    cols[f'hip_yaw_{s}'] = R(-p_yaw)                                 # feet keep pointing along the heading
    cols[f'shoulder_pitch_{s}'] = R(rl(shd)); cols[f'elbow_{s}'] = R(rl(elb))
cols['waist_yaw'] = R(t_yaw); cols['waist_roll'] = R(t_roll)        # from mocap: shoulders counter-rotate against the pelvis
names = list(cols); table = np.stack([cols[k] for k in names], 1)
J = hy.JOINTS; idx = np.array([J.index(k) for k in names])
w = np.array([1.0 if any(t in k for t in ('hip_pitch', 'knee', 'ankle_pitch')) else 0.3 if any(t in k for t in ('shoulder', 'elbow')) else 0.5 for k in names])
root = np.stack([R(p_roll), R(p_pitch), R(p_yaw), p_lat, p_up], 1)  # roll, pitch, yaw, lateral, vertical

# robot pelvis height of the reference pose (lowest sole on the ground), averaged over the cycle
m, qadr, dadr = sc.load(f'{sc.HERE}/model/hydro_humanoid.xml'); q0, _ = sc.default_q(m, qadr); qadr = np.array(qadr)
d = mujoco.MjData(m); sole = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'sole_{s}') for s in 'LR']
def low_sole():
    return min(min((d.geom_xpos[gi] + d.geom_xmat[gi].reshape(3, 3) @ (np.array([sx, sy, -1.0]) * m.geom_size[gi]))[2]
                   for sx in (-1, 1) for sy in (-1, 1)) for gi in sole)
hz = []
for i in range(n):
    qp = q0.copy(); qp[qadr[idx]] = table[i]; d.qpos[:] = qp; mujoco.mj_forward(m, d); hz.append(d.qpos[2] - low_sole())
f_h = 1.0 / float(z['stride_T']); v_h = float(z['speed_leglen']) * L_h
f_r = f_h * np.sqrt(L_h / L_r); v_r = v_h * np.sqrt(L_r / L_h)        # Froude scaling
np.savez(out, names=np.array(names), idx=idx, w=w.astype(np.float32), table=table.astype(np.float32), root=root.astype(np.float32),
         z_ref=float(np.mean(hz)), f_ref=f_r, v_ref=v_r)
print(f'{n} phases x {len(names)} joints | robot stride {1 / f_r:.2f}s ({f_r:.2f} Hz) at {v_r:.2f} m/s | pelvis height {np.mean(hz):.3f} m')
print(f'upper body: thorax twist {np.ptp(t_yaw):.1f} deg p-p (net shoulders vs heading {np.ptp(t_yaw + p_yaw):.1f}), arm swing {np.ptp(shd):.1f} deg p-p')
print(f'pelvis: roll {np.ptp(root[:, 0]) * 57.3:.1f} deg p-p, pitch {np.ptp(root[:, 1]) * 57.3:.1f}, yaw {np.ptp(root[:, 2]) * 57.3:.1f}, '
      f'sway {np.ptp(root[:, 3]) * 100:.1f} cm, bob {np.ptp(root[:, 4]) * 100:.1f} cm')
for k in range(len(names)): print(f'  {names[k]:18s} {np.degrees(table[:, k]).min():6.1f} .. {np.degrees(table[:, k]).max():6.1f} deg  (w {w[k]})')
