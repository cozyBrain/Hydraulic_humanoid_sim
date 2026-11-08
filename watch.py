"""Watch training robots LIVE while train_walk.py runs (separate terminal, costs the trainer nothing).
usage:  python watch.py [runs/walk]
keys:   A = all robots side by side / single robot   1..4 = which robot (single mode)   F = camera follow on/off   Space = pause
The trainer streams a few robots every 2 iterations; playback runs at the rate the data arrives (slow motion, since
training simulates slower than real time) about 1 s behind the newest data.
Shown motion = the actual training rollouts (exploration noise included), not a cleaned-up evaluation."""
import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # tools run on CPU: the GPU belongs to the trainer
import sys, os, time, numpy as np, mujoco, mujoco.viewer
import sim_core as sc, viz_util as vu
run = sys.argv[1] if len(sys.argv) > 1 else 'runs/walk'; LIVE = os.path.join(run, 'live.npz')
m = mujoco.MjModel.from_xml_path(f'{sc.HERE}/model/hydro_humanoid_viz.xml'); d = mujoco.MjData(m)
poser = vu.LinkagePoser(m); pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
qpos0, _ = sc.default_q(*sc.load(f'{sc.HERE}/model/hydro_humanoid_viz.xml')[0:2]); d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
LANE = 2.5                                                   # m between robots in the side-by-side view
d2 = mujoco.MjData(m); opt2 = mujoco.MjvOption(); opt2.geomgroup[:] = 0; opt2.geomgroup[1] = 1   # copies: robot meshes only (no 2nd floor)
pert2 = mujoco.MjvPerturb()
st = dict(all=True, env=0, follow=True, pause=False, f=None, pump=0.0, rate=None, last_end=None, t_end=None, acc=0.0)
def key(k):
    if ord('1') <= k <= ord('9'): st['env'] = k - ord('1'); st['f'] = None
    elif k == ord('F'): st['follow'] = not st['follow']
    elif k == ord('A'): st['all'] = not st['all']
    elif k == 32: st['pause'] = not st['pause']
buf, mtime = None, 0
with mujoco.viewer.launch_passive(m, d, key_callback=key) as v:
    v.cam.distance = 3.5; v.cam.elevation = -12; v.cam.azimuth = 135
    while v.is_running():
        t0 = time.time()
        try:
            mt = os.path.getmtime(LIVE)
            if mt != mtime:
                z = np.load(LIVE); buf = {k: z[k] for k in z.files}; mtime = mt
        except Exception: pass
        msg = 'waiting for training data (%s) ...' % LIVE
        if buf is not None:
            q, x, meta = buf['q'], buf['x'], buf['meta']; end = int(buf['frame_end']); start = end - len(q)
            e = min(st['env'], q.shape[1] - 1)
            # training produces simulated time slower than real time (each iteration = 0.4 s of robot time but
            # takes seconds of GPU time). Play back at the SAME rate the data arrives -> smooth slow motion, no freeze.
            now = time.time()
            if st['last_end'] is not None and end > st['last_end']:
                r = (end - st['last_end']) / max(now - st['t_end'], 1e-3)          # frames per wall-second arriving
                st['rate'] = r if st['rate'] is None else 0.7 * st['rate'] + 0.3 * r
            if st['last_end'] != end: st['last_end'], st['t_end'] = end, now
            if st['f'] is None or st['f'] < start: st['f'] = max(start, end - 60)
            rate = st['rate'] or 10.0
            lag_frames = end - 1 - st['f']
            rate *= 1.0 + 0.5 * np.clip((lag_frames - 40) / 40, -1, 1)          # drift toward ~0.8 s behind the newest data
            moved = False
            if not st['pause']:
                st['acc'] += rate * 0.02
                while st['acc'] >= 1 and st['f'] < end - 1: st['f'] += 1; st['acc'] -= 1; moved = True
                if st['f'] >= end - 1: st['acc'] = 0
            i = st['f'] - start
            ne = q.shape[1]
            if st['all']:                                        # robot 1 is the "real" one, the others are drawn as extra geoms
                lane = lambda k_: (k_ - (ne - 1) / 2) * LANE
                d.qpos[:] = q[i, 0]; d.qpos[1] += lane(0); mujoco.mj_forward(m, d)
            else:
                d.qpos[:] = q[i, e]; mujoco.mj_forward(m, d)
            Ps, w, P = x[i, e]
            if moved: st['pump'] += w * 0.02 * min(rate / 50.0, 1.0)        # gears advance with the robot's own (slowed) time
            poser(d, st['pump'])
            with v.lock():
                v.user_scn.ngeom = 0
                if st['all']:
                    for k_ in range(1, ne):
                        d2.qpos[:] = q[i, k_]; d2.qpos[1] += lane(k_); mujoco.mj_forward(m, d2); poser(d2, st['pump']); mujoco.mj_kinematics(m, d2)
                        mujoco.mjv_addGeoms(m, d2, opt2, pert2, mujoco.mjtCatBit.mjCAT_ALL, v.user_scn)
            if st['follow']:
                if st['all']: v.cam.lookat[:] = [np.mean(q[i, :, 0]), 0.0, 0.9]; v.cam.distance = max(v.cam.distance, 2.0 + LANE * ne * 0.9)
                else: v.cam.lookat[:] = d.xpos[pelvis]
            lag = (end - 1 - st['f']) * 0.02
            msg = ('stage %s   iteration %d   %.2e steps   mean return %.1f\n%s   Ps %.0f bar   pump %.0f rpm   power %.1f kW   playback %.0f%% of real time (training speed)%s' %
                   (str(buf['stage']), meta[i, 0], meta[i, 1], meta[i, 2], ('all %d robots (A = single)' % ne) if st['all'] else ('robot %d (A = all)' % (e + 1)), Ps, w * 9.549, P / 1e3, 100 * rate / 50, '   [PAUSED]' if st['pause'] else ''))
        try: v.set_texts([(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, msg.split('\n')[0], msg.split('\n')[1] if '\n' in msg else '')])
        except Exception: pass
        v.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
