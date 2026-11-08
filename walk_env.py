"""Walking environment for the v9 hydraulic humanoid (MJX + Brax API).

Control stack (as on the real robot):
  policy (50 Hz)  ->  joint position targets (offsets from the default crouch)
  firmware (1 kHz) ->  joint impedance -> actuator force targets (J^-T) -> pressure-feedback servo-valve command
  hydraulics (1 kHz) -> chamber pressures -> forces through the exact linkage kinematics -> MuJoCo
Observations are only what the robot can measure (IMU, encoders, chamber pressures, supply pressure) + noise.
The critic additionally sees privileged state (true base velocity, contacts, hydraulic internals)."""
import os, numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from brax.envs.base import Env, State
import hydraulics as hy, sim_core as sc

class HydroWalk(Env):
    def __init__(self, xml=None, ctrl_dt=0.02, action_scale=0.4, cmd_ranges=((-0.3, 0.8), (-0.3, 0.3), (-0.6, 0.6)), mode='full'):
        """mode='full': complete hydraulics at 1 ms.  mode='fast': reduced-order actuator model at 4 ms (stage-1 training)."""
        m, qadr, dadr = sc.load(xml)
        self.mode = mode
        if mode == 'fast': m.opt.timestep = 0.004
        self._m = m; self.sys = mjx.put_model(m)
        self.qadr, self.dadr = jnp.array(qadr), jnp.array(dadr)
        self.n_sub = int(round(ctrl_dt / m.opt.timestep)); self.ctrl_dt = ctrl_dt
        self.P = {k: jnp.array(v) for k, v in hy.default_params().items()}
        qpos0, q0 = sc.default_q(m, qadr); self.qpos0 = jnp.array(qpos0); self.q0 = jnp.array(q0)
        self.kp, self.kd = jnp.array(sc.KP), jnp.array(sc.KD)
        self.action_scale = action_scale; self.cmd_lo = jnp.array([c[0] for c in cmd_ranges]); self.cmd_hi = jnp.array([c[1] for c in cmd_ranges])
        jl = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in hy.JOINTS])
        self.jlo, self.jhi = jnp.array(jl[:, 0]), jnp.array(jl[:, 1])
        self.sole = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f'sole_{s}') for s in 'LR'])
        self.foot_body = jnp.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f'foot_{s}') for s in 'LR'])
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso')
        self.h0 = hy.init_state(self.P, q0) if mode == 'full' else hy.fast_init(self.P)
        self.z0 = float(qpos0[2])
        # ---- motion-capture imitation reference (CMU subject 35 walking, retargeted: mocap/make_ref.py)
        G = np.load(f'{sc.HERE}/gait_ref.npz')
        self.ref_tab = jnp.array(G['table']); self.ref_idx = jnp.array(G['idx'])
        self.ref_mean = jnp.mean(self.ref_tab, axis=0)
        self.ref_f = float(G['f_ref']); self.ref_v = float(G['v_ref']); self.ref_z = float(G['z_ref'])
        self.ref_w = jnp.array(G['w']); self.ref_root = jnp.array(G['root'])         # pelvis roll, pitch(+ nose up), yaw, sway, bob

    # ------------------------------------------------------------------ sizes
    @property
    def observation_size(self): return {'state': (12 + 4 * hy.NA,), 'privileged_state': (12 + 4 * hy.NA + 9 + 2 * hy.NA,)}
    @property
    def action_size(self): return hy.NA
    @property
    def backend(self): return 'mjx'

    # ------------------------------------------------------------------ helpers
    def _sample_cmd(self, rng):
        c = jax.random.uniform(rng, (3,), minval=self.cmd_lo, maxval=self.cmd_hi)
        k = jax.random.bernoulli(jax.random.fold_in(rng, 7), 0.1)            # 10 %: stand still
        return jnp.where(k, 0.0, c)
    def phase_freq(self, cmd):
        """gait-cycle frequency (Hz) for a command: human cadence-speed relation (f ~ sqrt(v)), mocap-scaled"""
        vx = cmd[0]
        return jnp.where(vx > 0.05, self.ref_f * jnp.sqrt(jnp.clip(vx / self.ref_v, 0.25, 1.2)), 0.9)
    def ref_pose(self, phase, cmd):
        """reference joint angles (ref_idx joints) at gait phase [rad], amplitude scaled down at lower speeds"""
        n = self.ref_tab.shape[0]; x = jnp.mod(phase / (2 * jnp.pi), 1.0) * (n - 1)
        i0 = jnp.floor(x).astype(jnp.int32); a = x - i0; i1 = jnp.minimum(i0 + 1, n - 1)
        q = self.ref_tab[i0] * (1 - a) + self.ref_tab[i1] * a
        r = self.ref_root[i0] * (1 - a) + self.ref_root[i1] * a
        s = jnp.sqrt(jnp.clip(cmd[0] / self.ref_v, 0.25, 1.0))
        return self.ref_mean + s * (q - self.ref_mean), s * r
    def _foot_contact(self, d):
        z = []
        for i in range(2):
            g = self.sole[i]; R = d.geom_xmat[g]; hs = self.sys.geom_size[g]
            corners = jnp.array([[sx, sy, -1.0] for sx in (-1, 1) for sy in (-1, 1)]) * hs
            z.append(jnp.min(d.geom_xpos[g][2] + corners @ R[2]))
        z = jnp.array(z); return z < 0.006, z

    def _obs(self, d, h, info, rng):
        obs = obs_core(self, d.qpos, d.qvel, h, info, rng)
        R = _quat2mat(d.qpos[3:7])
        contact, _ = self._foot_contact(d)
        lin = R.T @ d.qvel[0:3]
        if self.mode == 'full': hyd = jnp.concatenate([(h['pA'] - h['pB']) / 2e7, h['xv']])
        else:
            Fs = self.P['Ps_set'] * self.P['AA']; hyd = jnp.concatenate([h['F'] / Fs, jnp.clip(h['Fdes'] / Fs, -1.5, 1.5)])
        priv = jnp.concatenate([obs, lin, contact.astype(jnp.float32), hyd, jnp.array([d.qpos[2] - self.z0, (h['T'] - 40) / 20, h['Voil'] * 1e3, info['push_t']])])
        return {'state': obs, 'privileged_state': priv}

    # ------------------------------------------------------------------ API
    def reset(self, rng):
        k = jax.random.split(rng, 6)
        qpos = self.qpos0.at[self.qadr].add(jax.random.uniform(k[0], (hy.NA,), minval=-0.05, maxval=0.05))
        qvel = jnp.zeros(self._m.nv).at[0:2].set(jax.random.uniform(k[1], (2,), minval=-0.1, maxval=0.1))
        d = mjx.make_data(self.sys).replace(qpos=qpos, qvel=qvel); d = mjx.forward(self.sys, d)
        hscale = 1.0 + jax.random.uniform(k[2], (5,), minval=-0.15, maxval=0.15)     # beta, Kv, leakage, friction, valve lag
        info = dict(cmd=self._sample_cmd(k[3]), last_act=jnp.zeros(hy.NA), last_last_act=jnp.zeros(hy.NA), phase=jnp.zeros((), jnp.float32),
                    h=self.h0, hscale=hscale, Ps=jnp.full((), 2.07e7, jnp.float32), step=jnp.zeros((), jnp.int32), air=jnp.zeros(2), last_contact=jnp.ones(2, bool),
                    push=jnp.zeros(3), push_t=jnp.zeros((), jnp.float32), rng=k[4], E0=jnp.zeros((), jnp.float32), head_err=jnp.zeros((), jnp.float32), wz_f=jnp.zeros((), jnp.float32), foot_xy=d.xpos[self.foot_body][:, :2])
        obs = self._obs(d, self.h0, info, k[5])
        metrics = {k_: jnp.zeros((), jnp.float32) for k_ in ('r_imit', 'r_vel', 'r_yaw', 'p_turn', 'r_up', 'r_height', 'r_air', 'p_power', 'p_act', 'p_slip', 'p_lim', 'p_arm', 'power_W', 'Ps_bar')}
        return State(d, obs, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32), metrics, info)

    def step(self, state, action):
        info = dict(state.info); rng, k_push, k_cmd, k_obs = jax.random.split(info['rng'], 4)
        P = dict(self.P); s5 = info['hscale']
        P['beta'] = P['beta'] * s5[0]; P['Kv'] = P['Kv'] * s5[1]; P['Cl'] = P['Cl'] * s5[2]; P['Fc'] = P['Fc'] * s5[3]; P['tau_v'] = P['tau_v'] * s5[4]
        a = jnp.clip(action, -1, 1)
        q_tgt = jnp.clip(self.q0 + a * self.action_scale, self.jlo, self.jhi)
        # random pushes on the torso (every ~3 s, 0.1 s long)
        push_now = jax.random.bernoulli(k_push, self.ctrl_dt / 3.0)
        push = jnp.where(push_now, jax.random.normal(jax.random.fold_in(k_push, 1), (3,)) * jnp.array([120.0, 120.0, 30.0]), info['push'])
        push_t = jnp.where(push_now, 0.1, jnp.maximum(info['push_t'] - self.ctrl_dt, 0.0)); push = jnp.where(push_t > 0, push, 0.0)
        dt = self._m.opt.timestep
        def sub(carry, _):
            d, h = carry
            q = d.qpos[self.qadr]; qd = d.qvel[self.dadr]
            tau_des = self.kp * (q_tgt - q) - self.kd * qd
            xfrc = jnp.zeros((self._m.nbody, 6)).at[self.torso, 0:3].set(push)
            if self.mode == 'full':
                tau, F, J, v = hy.forces(P, h, q, qd)
                u = hy.servo_command(P, h, q, qd, tau_des, J, v)
                d = d.replace(qfrc_applied=jnp.zeros(self._m.nv).at[self.dadr].set(tau), xfrc_applied=xfrc)
                d = mjx.step(self.sys, d)
                h, _, diag = hy.step(P, h, d.qpos[self.qadr], d.qvel[self.dadr], u, jnp.zeros(hy.NA), dt)
            else:
                tau, F, J, v = hy.fast_forces(P, h, q, qd)
                d = d.replace(qfrc_applied=jnp.zeros(self._m.nv).at[self.dadr].set(tau), xfrc_applied=xfrc)
                d = mjx.step(self.sys, d)
                h, diag = hy.fast_step(P, h, d.qpos[self.qadr], d.qvel[self.dadr], tau_des, dt)
            return (d, h), (diag['Pel'], diag['Ps'])
        (d, h), (Pel, Ps) = jax.lax.scan(sub, (state.pipeline_state, info['h']), None, length=self.n_sub)
        power = jnp.mean(Pel)
        # ---------------- rewards
        R = _quat2mat(d.qpos[3:7]); lin = R.T @ d.qvel[0:3]; ang = d.qvel[3:6]; cmd = info['cmd']
        contact, footz = self._foot_contact(d)
        ev = jnp.sum((cmd[:2] - lin[:2]) ** 2)
        r_vel = 0.6 * jnp.exp(-ev / 0.1) + 0.4 * jnp.exp(-ev / 0.6)              # wide part: still pays for half-way progress
        r_yaw = jnp.exp(-(cmd[2] - ang[2]) ** 2 / 0.2)
        up = (R.T @ jnp.array([0.0, 0, 1]))[:2]; r_up = jnp.exp(-jnp.sum(up ** 2) / 0.02)
        r_height = jnp.exp(-((d.qpos[2] - self.z0) / 0.06) ** 2)
        first = contact & ~info['last_contact']; air = info['air'] + self.ctrl_dt
        moving = jnp.linalg.norm(cmd) > 0.1
        r_air = jnp.sum((jnp.minimum(air, 0.5) - 0.25) * first) * moving
        air = jnp.where(contact, 0.0, air)
        foot_xy = d.xpos[self.foot_body][:, :2]
        foot_v = (foot_xy - info['foot_xy']) / self.ctrl_dt                           # finite-difference foot velocity
        p_slip = jnp.sum(jnp.sum(foot_v ** 2, axis=1) * contact)
        p_act = jnp.sum((a - info['last_act']) ** 2) + 0.5 * jnp.sum((a - 2 * info['last_act'] + info['last_last_act']) ** 2)
        q = d.qpos[self.qadr]; margin = 0.05
        p_lim = jnp.sum(jnp.maximum(self.jlo + margin - q, 0) + jnp.maximum(q - self.jhi + margin, 0))
        p_power = power / 1000.0
        cmd_sp = jnp.linalg.norm(cmd[:2]); p_stall = jnp.maximum(0.6 * cmd_sp - jnp.dot(lin[:2], cmd[:2]) / (cmd_sp + 1e-6), 0.0) * (cmd_sp > 0.1)   # told to go, but not going
        stand_still = jnp.where(moving, 0.0, jnp.sum((q - self.q0) ** 2))
        p_arm = jnp.sum((q[hy.ARM0:] - self.q0[hy.ARM0:]) ** 2)               # arms may swing for balance, but no flailing
        fy = (R.T @ (d.xpos[self.foot_body[0]] - d.xpos[self.foot_body[1]]))[1]
        p_cross = jnp.maximum(0.11 - fy, 0.0) * 10.0                            # training model has no self-collision: keep the feet apart
        # turning off-command: linear like p_stall (exp-shaped r_yaw is ~free for small drift)
        wz = (R @ ang)[2]                                                      # world-frame yaw rate
        wz_f = info['wz_f'] + (wz - info['wz_f']) * self.ctrl_dt / 1.0         # ~1 s average: ignores the natural +-5 deg pelvis twist each step
        p_turn = jnp.minimum(jnp.abs(cmd[2] - wz_f), 0.5)                      # rad/s off the commanded turn rate, capped (max = p_stall max)
        head_err = jnp.clip(info['head_err'] + (wz - cmd[2]) * self.ctrl_dt, -0.6, 0.6)   # heading drifted away from the command [rad]
        p_head = jnp.maximum(jnp.abs(head_err) - 0.1, 0.0) * moving            # slow drift the rate term misses (0.1 rad dead zone for the pelvis twist)
        gi = jnp.clip(cmd[0] / 0.15, 0.0, 1.0) * moving                      # imitation only while walking forward
        qr, rr = self.ref_pose(info['phase'], cmd); e2 = (q[self.ref_idx] - qr) ** 2
        r_imit = jnp.exp(-jnp.sum(e2 * self.ref_w) / 0.3)
        p_roll = jnp.arctan2(R[2, 1], R[2, 2]); p_pitch_up = jnp.arcsin(jnp.clip(R[2, 0], -1, 1))       # pelvis roll (+ left up), pitch (+ nose up)
        r_pel = jnp.exp(-((p_roll - rr[0]) ** 2 + (p_pitch_up - rr[1]) ** 2) / 0.005)                  # pelvis obliquity / tilt like a human
        r_height = jnp.exp(-((d.qpos[2] - (self.z0 + gi * (self.ref_z - self.z0 + rr[4]))) / 0.06) ** 2)
        reward = (1.5 * gi * r_imit + 0.5 * gi * r_pel + 1.5 * r_vel + 0.8 * r_yaw + 0.5 * r_up + 0.3 * r_height + 1.0 * r_air
                  - 0.05 * p_power - 0.02 * p_act - 0.1 * p_slip - 2.0 * p_lim - 0.2 * stand_still - 0.1 * (1 - gi) * p_arm - 1.0 * p_cross - 2.0 * p_stall - 2.0 * p_turn - 2.0 * p_head + 0.2) * self.ctrl_dt * 50
        fell = (d.qpos[2] < self.z0 - 0.35) | (jnp.sum(up ** 2) > 0.5) | jnp.any(jnp.isnan(d.qpos))
        reward = jnp.where(fell, -10.0, reward)          # (was -25: so scary that the policy froze after one step)
        # ---------------- bookkeeping
        step = info['step'] + 1
        newcmd = jnp.where(step % 250 == 0, self._sample_cmd(k_cmd), cmd)
        info.update(cmd=newcmd, wz_f=wz_f, head_err=jnp.where(step % 250 == 0, 0.0, head_err), last_last_act=info['last_act'], last_act=a, phase=jnp.mod(info['phase'] + 2 * jnp.pi * self.phase_freq(newcmd) * self.ctrl_dt, 2 * jnp.pi),
                    h=h, Ps=Ps[-1], step=step, foot_xy=foot_xy, air=air, last_contact=contact, push=push, push_t=push_t, rng=rng)
        obs = self._obs(d, h, info, k_obs)
        metrics = dict(state.metrics); metrics.update(r_imit=r_imit * gi, r_vel=r_vel, r_yaw=r_yaw, p_turn=p_turn, r_up=r_up, r_height=r_height, r_air=r_air, p_power=p_power, p_act=p_act, p_slip=p_slip, p_lim=p_lim, p_arm=p_arm,
                       power_W=power, Ps_bar=Ps[-1] / 1e5)
        done = fell.astype(jnp.float32)
        obs = jax.tree.map(lambda x: jnp.nan_to_num(x), obs)
        return state.replace(pipeline_state=d, obs=obs, reward=reward, done=done, metrics=metrics, info=info)

def _quat2mat(q):
    w, x, y, z = q
    return jnp.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def domain_randomize(mx, rng):
    """per-env model randomisation (Brax randomization_fn): floor/foot friction, torso mass, joint damping"""
    @jax.vmap
    def rand(key):
        k = jax.random.split(key, 3)
        fr = mx.geom_friction.at[:, 0].set(jax.random.uniform(k[0], minval=0.5, maxval=1.25))
        bm = mx.body_mass * jnp.where(jnp.arange(mx.nbody) > 0, jax.random.uniform(k[1], (mx.nbody,), minval=0.9, maxval=1.1), 1.0)
        dmp = mx.dof_damping * jax.random.uniform(k[2], (mx.nv,), minval=0.7, maxval=1.3)
        return fr, bm, dmp
    fr, bm, dmp = rand(rng)
    axes = jax.tree.map(lambda x: None, mx).tree_replace({'geom_friction': 0, 'body_mass': 0, 'dof_damping': 0})
    mx = mx.tree_replace({'geom_friction': fr, 'body_mass': bm, 'dof_damping': dmp})
    return mx, axes

def obs_core(env, qpos, qvel, h, info, rng=None):
    """policy observation: only on-robot sensors (IMU, encoders, chamber + supply pressure) + command, phase, last action"""
    q = qpos[env.qadr]; qd = qvel[env.dadr]
    R = _quat2mat(qpos[3:7]); grav = R.T @ jnp.array([0.0, 0, -1]); gyro = qvel[3:6]
    F = (h['pA'] * env.P['AA'] - h['pB'] * env.P['AB']) if 'pA' in h else h['F']      # pressure sensors (full) / actuator force (fast)
    _, J = hy.actuator_jac(q); tau_est = J.T @ F
    if rng is None: nz = [jnp.zeros(3), jnp.zeros(3), jnp.zeros(hy.NA), jnp.zeros(hy.NA), jnp.zeros(hy.NA)]
    else:
        k = jax.random.split(rng, 5)
        nz = [jax.random.normal(k[0], (3,)) * 0.03, jax.random.normal(k[1], (3,)) * 0.15, jax.random.normal(k[2], (hy.NA,)) * 0.01,
              jax.random.normal(k[3], (hy.NA,)) * 0.1, jax.random.normal(k[4], (hy.NA,)) * 0.02]
    phase = info['phase']
    return jnp.concatenate([grav + nz[0], gyro + nz[1], (q - env.q0) + nz[2], qd * 0.1 + nz[3], tau_est / 200.0 + nz[4],
                            info['last_act'], info['cmd'], jnp.array([jnp.sin(phase), jnp.cos(phase)]), jnp.array([(info['Ps'] - 2.0e7) / 2e6])])
