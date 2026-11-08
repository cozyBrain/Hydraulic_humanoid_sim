"""MuJoCo-MJX + hydraulics coupling.  One physics step = 1 ms:
   hydraulic forces (from chamber pressures) -> qfrc_applied -> mjx.step -> pressures integrated with the new velocity."""
import os, numpy as np, mujoco, jax, jax.numpy as jnp
from mujoco import mjx
import hydraulics as hy
HERE = os.path.dirname(os.path.abspath(__file__))

def load(xml=None):
    m = mujoco.MjModel.from_xml_path(xml or f'{HERE}/model/hydro_humanoid.xml')
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in hy.JOINTS]
    qadr = np.array([m.jnt_qposadr[j] for j in jid]); dadr = np.array([m.jnt_dofadr[j] for j in jid])
    m.dof_armature[dadr] += hy.fluid_armature({k: jnp.array(v) for k, v in hy.default_params().items()}, np.zeros(hy.NA))
    return m, qadr, dadr

DEFAULT_POSE = dict(hip_pitch=12.0, knee=30.0, ankle_pitch=-18.0, waist_pitch=6.0, elbow=15.0, shoulder_roll=4.0)     # mild crouch, feet flat, COM over mid-foot
def default_q(m, qadr):
    q = np.zeros(hy.NA)
    for i, n in enumerate(hy.JOINTS):
        for k, v in DEFAULT_POSE.items():
            if n.startswith(k): q[i] = np.radians(v)
    d = mujoco.MjData(m); d.qpos[qadr] = q; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[2] = 1.0
    mujoco.mj_kinematics(m, d)
    # drop so the lowest sole corner touches the floor
    zmin = 9
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0: continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            R = d.geom_xmat[g].reshape(3, 3); hs = m.geom_size[g]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1): zmin = min(zmin, (d.geom_xpos[g] + R @ (hs * [sx, sy, sz]))[2])
    qpos = d.qpos.copy(); qpos[2] -= zmin - 0.0005
    return qpos, q

def make_step(mx, qadr, dadr, P):
    qadr, dadr = jnp.array(qadr), jnp.array(dadr)
    def phys_step(d, h, u_spool, u_byp, motor_on=1.0):
        tau, _, _, _ = hy.forces(P, h, d.qpos[qadr], d.qvel[dadr])
        d = d.replace(qfrc_applied=jnp.zeros_like(d.qfrc_applied).at[dadr].set(tau))
        d = mjx.step(mx, d)
        # symplectic coupling: pressures integrate with the post-step velocity
        hn, _, diag = hy.step(P, h, d.qpos[qadr], d.qvel[dadr], u_spool, u_byp, mx.opt.timestep, motor_on)
        return d, hn, diag
    return phys_step

def make_servo(mx, qadr, dadr, P):
    qadr, dadr = jnp.array(qadr), jnp.array(dadr)
    def servo(d, h, q_tgt, kp, kd, tau_ff=0.0):
        q = d.qpos[qadr]; qd = d.qvel[dadr]
        x, J = hy.actuator_jac(q); v = J @ qd
        tau_des = kp * (q_tgt - q) - kd * qd + tau_ff
        return hy.servo_command(P, h, q, qd, tau_des, J, v), tau_des
    return servo

# joint-space impedance gains for the firmware (N m / rad, N m s / rad)
KP = np.array([300, 600, 500] + [250, 600, 600, 600, 900, 300] * 2 + [200, 200, 60, 120, 20, 20, 15] * 2, np.float32)
KD = np.array([10, 20, 18] + [8, 20, 20, 15, 10, 4] * 2 + [8, 8, 2, 4, 0.8, 0.8, 0.6] * 2, np.float32)

# ---------------------------------------------------------------- CPU backend (C MuJoCo + jitted hydraulics), for viewer / tests
class CpuSim:
    def __init__(self, xml=None, hung=False, params=None):
        self.m, self.qadr, self.dadr = load(xml)
        if hung:   # pin the pelvis in the air (test bench): very stiff weld via equality would also work; here freeze root dofs
            self.m.dof_damping[:6] = 1e6; self.m.dof_armature[:6] = 1e3; self.m.body_gravcomp[1] = 0
        self.P = {k: jnp.array(v) for k, v in (params or hy.default_params()).items()}
        self.d = mujoco.MjData(self.m)
        qpos, self.q0 = default_q(self.m, self.qadr)
        if hung: qpos[2] += 0.3
        self.d.qpos[:] = qpos; mujoco.mj_forward(self.m, self.d)
        self.h = jax.tree.map(np.asarray, hy.init_state(self.P, self.q0))
        dt = self.m.opt.timestep; P = self.P
        def ctl(h, q, qd, q_tgt, kp, kd, spool_override, use_override):
            tau, F, J, v = hy.forces(P, h, q, qd)
            tau_des = kp * (q_tgt - q) - kd * qd
            u = hy.servo_command(P, h, q, qd, tau_des, J, v)
            return tau, jnp.where(use_override, spool_override, u)
        def hyd(h, q, qd, u, byp, motor_on):
            hn, _, diag = hy.step(P, h, q, qd, u, byp, dt, motor_on); return hn, diag
        self._c = jax.jit(ctl); self._h = jax.jit(hyd)
        self.kp, self.kd = jnp.array(KP), jnp.array(KD)
    def step(self, q_tgt, byp=None, motor_on=1.0, spool=None):
        d = self.d; q = d.qpos[self.qadr]; qd = d.qvel[self.dadr]
        tau, u = self._c(self.h, q, qd, jnp.asarray(q_tgt, jnp.float32), self.kp, self.kd,
                         jnp.zeros(hy.NA) if spool is None else jnp.asarray(spool), spool is not None)
        d.qfrc_applied[:] = 0; d.qfrc_applied[self.dadr] = np.asarray(tau)
        mujoco.mj_step(self.m, d)
        self.h, diag = self._h(self.h, d.qpos[self.qadr], d.qvel[self.dadr], u, jnp.zeros(hy.NA) if byp is None else jnp.asarray(byp), motor_on)
        return diag, u

def stand_targets(S):
    """demo balance controller for running WITHOUT a learned policy (uses simulator COM, i.e. privileged):
    ankle strategy keeps the COM over the feet, hip strategy keeps the pelvis upright."""
    m, d = S.m, S.d; J = hy.JOINTS; qt = S.q0.copy()
    if not hasattr(S, 'sole'): S.sole = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'sole_{s}') for s in 'LR']
    w, x, y, z = d.qpos[3:7]; pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)); roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    com = d.subtree_com[1]; fx = 0.5 * (d.geom_xpos[S.sole[0]] + d.geom_xpos[S.sole[1]]); v = d.qvel[0:3]
    for s in 'LR':
        qt[J.index(f'ankle_pitch_{s}')] += 4.0 * (com[0] - fx[0]) + 0.6 * v[0]
        qt[J.index(f'ankle_roll_{s}')] += -3.0 * (com[1] - fx[1]) - 0.4 * v[1]
        qt[J.index(f'hip_pitch_{s}')] += -0.8 * pitch - 0.05 * d.qvel[4]
        qt[J.index(f'hip_roll_{s}')] += 0.8 * roll + 0.05 * d.qvel[3]
    return qt
