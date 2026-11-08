"""Interactive viewer: full hydraulic simulation (1 kHz physics + hydraulics), optional trained policy (50 Hz).

usage:  python viewer.py                       # starts in STAND mode (demo balance controller)
        python viewer.py runs/walk/params.pkl  # policy available (key 3)

modes   1 = STAND   balance controller keeps the robot upright
        2 = MANUAL  you drive every joint with the sliders in the right-hand "Control" panel (degrees).
                    The sliders are joint TARGETS for the on-robot firmware -> servo valves -> oil pressure -> force.
                    Nothing is kinematically forced: if a joint is overloaded or the pump can't keep up, it lags.
        3 = POLICY  trained walking policy (arrows: speed / turn, , . : side step, space: stop)
keys    H = gantry on/off (holds the pelvis in the air, handy in MANUAL mode)
        G = slow motion  1x -> 1/4 -> 1/10 -> 1/40 -> 1/100  (whole simulation; pump gears need ~1/40 to follow the teeth)
        N = sound on/off  P = push torso    M = hydraulic power on/off (off: pump stops, valves centre, bypass opens -> limp)
        R = reset
Tip: the Control panel is the right-hand UI panel of the MuJoCo viewer (toggle panels with Tab / Shift+Tab)."""
import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # tools run on CPU: the GPU belongs to the trainer
import sys, time, pickle, numpy as np, mujoco, mujoco.viewer, jax, jax.numpy as jnp
import hydraulics as hy, sim_core as sc, viz_util as vu

XML = f'{sc.HERE}/model/hydro_humanoid_viz.xml'
S = sc.CpuSim(XML); m, d = S.m, S.d
poser = vu.LinkagePoser(m)
torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso'); pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
act_of_joint = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hy.JOINTS])   # slider index per joint
SLOW = [1, 4, 10, 40, 100]
st = dict(cmd=np.zeros(3), motor=1.0, push=0, reset=False, mode=1, slow=0, gantry=False, g_pos=None, g_quat=None, pump=0.0, acc=0.0)

policy = None
if len(sys.argv) > 1:
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from walk_env import HydroWalk, obs_core
    env = HydroWalk()
    nets = ppo_networks.make_ppo_networks(env.observation_size, hy.NA, preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 256, 128),
                                          policy_obs_key='state', value_obs_key='privileged_state')
    policy = jax.jit(ppo_networks.make_inference_fn(nets)(pickle.load(open(sys.argv[1], 'rb')), deterministic=True))
    obs_fn = jax.jit(lambda qpos, qvel, h, info: obs_core(env, qpos, qvel, h, info))

def key(k):
    c = st['cmd']
    if k == ord('1'): st['mode'] = 1
    elif k == ord('2'): st['mode'] = 2
    elif k == ord('3') and policy is not None: st['mode'] = 3
    elif k == 265: c[0] = min(c[0] + 0.1, 0.8)
    elif k == 264: c[0] = max(c[0] - 0.1, -0.3)
    elif k == 263: c[2] = min(c[2] + 0.15, 0.6)
    elif k == 262: c[2] = max(c[2] - 0.15, -0.6)
    elif k == ord(','): c[1] = min(c[1] + 0.1, 0.3)
    elif k == ord('.'): c[1] = max(c[1] - 0.1, -0.3)
    elif k == 32: c[:] = 0
    elif k == ord('P'): st['push'] = 100
    elif k == ord('M'): st['motor'] = 1.0 - st['motor']
    elif k == ord('N') and SND is not None: print('[sound]', 'muted' if SND.toggle() else 'on')
    elif k == ord('R'): st['reset'] = True
    elif k == ord('G'): st['slow'] = (st['slow'] + 1) % len(SLOW)
    elif k == ord('H'):
        st['gantry'] = not st['gantry']
        if st['gantry']: st['g_pos'] = d.xpos[pelvis].copy() + [0, 0, 0.15]; st['g_quat'] = np.array([1.0, 0, 0, 0])
        # the crane is a heavy fixture: extra root inertia keeps its explicit spring-damper numerically stable
        # (without it the pelvis alone (~0.05 kg m^2) against 400 Nm s/rad damping at 1 ms blew up the sim)
        m.dof_armature[3:6] = GANTRY_ARM if st['gantry'] else ROOT_ARM

def reset():
    qpos, _ = sc.default_q(m, S.qadr); d.qpos[:] = qpos; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    S.h = jax.tree.map(np.asarray, hy.init_state(S.P, S.q0))
    d.ctrl[act_of_joint] = np.degrees(S.q0)
    return dict(cmd=jnp.zeros(3), last_act=jnp.zeros(hy.NA), phase=jnp.array(0.0), Ps=jnp.array(2.07e7))

ROOT_ARM = m.dof_armature[3:6].copy(); GANTRY_ARM = 50.0
def gantry_force():
    """virtual crane: stiff spring-damper holding the pelvis pose (applied as an external wrench)"""
    p, v = d.xpos[pelvis], d.cvel[pelvis]
    F = 30000 * (st['g_pos'] - p) - 3000 * d.qvel[0:3]
    q = d.xquat[pelvis]; qe = np.zeros(4); qi = np.zeros(4); mujoco.mju_negQuat(qi, q); mujoco.mju_mulQuat(qe, st['g_quat'], qi)
    rv = np.zeros(3); mujoco.mju_quat2Vel(rv, qe, 1.0)
    Tq = 4000 * rv - 400 * (d.xmat[pelvis].reshape(3, 3) @ d.qvel[3:6])       # free-joint ang. velocity is body-frame -> world
    return F, Tq

info = reset(); q_tgt = S.q0.copy(); t_ctrl = 0; diag = None
import sound; SND = sound.try_open()          # optional live audio (N = mute)
with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
    v.cam.distance = 3.2; v.cam.elevation = -12; v.cam.azimuth = 135
    while v.is_running():
        t0 = time.time()
        if st['reset']: info = reset(); st['reset'] = False
        st['acc'] += 10.0 / SLOW[st['slow']]                 # physics steps (1 ms) owed for this 10 ms display tick
        n = int(st['acc']); st['acc'] -= n
        for _ in range(n):
            mode = st['mode']
            if mode == 3 and t_ctrl % 20 == 0:
                info['cmd'] = jnp.array(st['cmd'], jnp.float32)
                obs = obs_fn(jnp.array(d.qpos), jnp.array(d.qvel), S.h, info)
                a, _ = policy({'state': obs, 'privileged_state': jnp.zeros(env.observation_size['privileged_state'])}, jax.random.PRNGKey(0))
                a = np.clip(np.array(a), -1, 1); q_tgt = np.clip(S.q0 + a * env.action_scale, np.array(env.jlo), np.array(env.jhi))
                info.update(last_act=jnp.array(a), phase=jnp.mod(info['phase'] + 2 * np.pi * float(env.phase_freq(info['cmd'])) * 0.02, 2 * np.pi))
            elif mode == 2: q_tgt = np.radians(d.ctrl[act_of_joint])
            elif mode == 1: q_tgt = sc.stand_targets(S)
            if mode != 2: d.ctrl[act_of_joint] = np.degrees(q_tgt)          # sliders follow the targets until you take over
            d.xfrc_applied[:] = 0
            if st['push'] > 0: d.xfrc_applied[torso, :3] = [150, 60, 0]; st['push'] -= 1
            if st['gantry']: F, Tq = gantry_force(); d.xfrc_applied[pelvis, :3] += F; d.xfrc_applied[pelvis, 3:] += Tq
            diag, u = S.step(q_tgt, byp=None, motor_on=st['motor']); t_ctrl += 1
            if not (np.isfinite(d.qacc).all() and np.isfinite(d.qpos).all()) or np.abs(d.qvel).max() > 1e3:
                # simulation blew up: print what was going on, then auto-reset instead of freezing the viewer
                q = d.qpos[S.qadr]
                print(f'\n[blow-up] t={d.time:.3f}s mode={mode}  max|qvel|={np.nanmax(np.abs(d.qvel)):.3g}  contacts={d.ncon}')
                for c in d.contact[:d.ncon]:
                    print('   contact', m.body(m.geom_bodyid[c.geom1]).name, '<->', m.body(m.geom_bodyid[c.geom2]).name, f'depth {-c.dist*1000:.1f} mm')
                print('   joints(deg):', {n: round(float(np.degrees(x)), 1) for n, x in zip(hy.JOINTS, q)})
                print('   hydraulics finite:', {k: bool(np.isfinite(np.asarray(x)).all()) for k, x in S.h.items()}, ' -> auto reset\n', flush=True)
                info = reset(); break
            info['Ps'] = diag['Ps']; st['pump'] += float(S.h['w']) * 0.001      # real pump shaft angle (sim time)
        poser(d, st['pump']); v.sync()
        if SND is not None and diag is not None:
            SND.update(rpm=float(S.h['w']) * 9.549, Ps=float(diag['Ps']) / 1e5, Pel=float(diag['Pel']), Q=float(diag['Qdraw']) * 60000,
                       slow=SLOW[st['slow']], on=st['motor'])
        if diag is not None:
            try:
                q = d.qpos[S.qadr]; err = np.degrees(np.abs(q_tgt - q)).max()
                l1 = 'Ps %.0f bar  pump %.0f rpm  P_el %.1f kW  I_bat %.0f A  oil %.1f C  acc %.0f mL  %s' % (
                    float(diag['Ps']) / 1e5, float(S.h['w']) * 9.549, float(diag['Pel']) / 1e3, float(diag['Ibat']), float(S.h['T']), float(S.h['Voil']) * 1e6,
                    'MOTOR ON' if st['motor'] else 'MOTOR OFF (limp)')
                l2 = 'mode %s   speed %s   gantry %s   max tracking error %.1f deg   cmd vx %.1f vy %.1f yaw %.2f' % (
                    {1: 'STAND', 2: 'MANUAL (sliders)', 3: 'POLICY'}[st['mode']], '1x' if SLOW[st['slow']] == 1 else '1/%d' % SLOW[st['slow']],
                    'ON' if st['gantry'] else 'off', err, *st['cmd'])
                v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, l1, l2)])
            except Exception: pass
        time.sleep(max(0.0, 0.01 - (time.time() - t0)))
