import sys, os, numpy as np, mujoco
from PIL import Image
import sim_core as sc, viz_util as vu
m, qadr, dadr = sc.load(f'{sc.HERE}/model/hydro_humanoid_viz.xml'); d = mujoco.MjData(m)
qpos, q0 = sc.default_q(m, qadr); d.qpos[:] = qpos
J = sc.hy.JOINTS
for kv in sys.argv[2:]:
    k, v = kv.split('='); d.qpos[qadr[J.index(k)]] = np.radians(float(v))
mujoco.mj_forward(m, d); vu.LinkagePoser(m)(d); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 900, 900); cam = mujoco.MjvCamera()
cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, os.environ.get('LOOK', 'shin_L'))]; cam.distance = float(os.environ.get('DIST', 1.1)); cam.azimuth = float(os.environ.get('AZ', 90)); cam.elevation = -5
r.update_scene(d, cam); Image.fromarray(r.render()).save(sys.argv[1]); print('ok')
