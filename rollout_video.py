"""Headless rollout -> mp4 (+ CSV of hydraulic telemetry).
usage: python rollout_video.py [params.pkl|none] out.mp4 [seconds] [vx] [yaw]"""
import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # tools run on CPU: the GPU belongs to the trainer
import sys, numpy as np, mujoco, jax, jax.numpy as jnp, imageio
import hydraulics as hy, sim_core as sc, viz_util as vu
pk = sys.argv[1] if len(sys.argv) > 1 else 'none'; out = sys.argv[2] if len(sys.argv) > 2 else 'rollout.mp4'
T = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0; vx = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5; yaw = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
S = sc.CpuSim(f'{sc.HERE}/model/hydro_humanoid_viz.xml'); m, d = S.m, S.d; poser = vu.LinkagePoser(m)
policy = None
if pk != 'none':
    import pickle
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from walk_env import HydroWalk, obs_core
    env = HydroWalk()
    nets = ppo_networks.make_ppo_networks(env.observation_size, hy.NA, preprocess_observations_fn=running_statistics.normalize,
                                          policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 256, 128),
                                          policy_obs_key='state', value_obs_key='privileged_state')
    policy = jax.jit(ppo_networks.make_inference_fn(nets)(pickle.load(open(pk, 'rb')), deterministic=True))
    obs_fn = jax.jit(lambda qpos, qvel, h, info: obs_core(env, qpos, qvel, h, info))
info = dict(cmd=jnp.array([vx, 0, yaw], jnp.float32), last_act=jnp.zeros(hy.NA), phase=jnp.array(0.0), Ps=jnp.array(2.07e7))
r = mujoco.Renderer(m, 720, 960); cam = mujoco.MjvCamera(); cam.distance = 3.4; cam.elevation = -10; cam.azimuth = 120
frames, rows, q_tgt = [], [], S.q0.copy(); pump_ang = 0.0
for k in range(int(T * 1000)):
    if policy is not None and k % 20 == 0:
        obs = obs_fn(jnp.array(d.qpos), jnp.array(d.qvel), S.h, info)
        a, _ = policy({'state': obs, 'privileged_state': jnp.zeros(env.observation_size['privileged_state'])}, jax.random.PRNGKey(0))
        a = np.clip(np.array(a), -1, 1); q_tgt = np.clip(S.q0 + a * env.action_scale, np.array(env.jlo), np.array(env.jhi))
        info.update(last_act=jnp.array(a), phase=jnp.mod(info['phase'] + 2 * np.pi * 1.6 * 0.02, 2 * np.pi))
    if policy is None: q_tgt = sc.stand_targets(S)
    d.xfrc_applied[:] = 0
    if pk == 'none' and 1500 <= k < 1600: d.xfrc_applied[S.m.nbody - 1 if False else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'torso'), :3] = [180, 0, 0]   # push test
    diag, u = S.step(q_tgt); info['Ps'] = diag['Ps']; pump_ang += float(S.h['w']) * 0.001
    if k % 10 == 0: rows.append([k / 1000, d.qpos[0], d.qpos[2], float(diag['Ps']) / 1e5, float(diag['Pel']), float(S.h['T'])] + list(np.degrees(d.qpos[S.qadr])))
    if k % 33 == 0:
        poser(d, pump_ang); mujoco.mj_forward(m, d); cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.85]
        r.update_scene(d, cam); frames.append(r.render())
    if d.qpos[2] < 0.3: break
imageio.mimsave(out, frames, fps=30, quality=8)
np.savetxt(out.replace('.mp4', '.csv'), np.array(rows), delimiter=',', fmt='%.4f',
           header='t,x,z,Ps_bar,Pel_W,oil_C,' + ','.join(hy.JOINTS), comments='')
print('wrote', out, len(frames), 'frames; final x %.2f m, z %.2f m' % (d.qpos[0], d.qpos[2]))
