import mujoco, numpy as np, sys, os
from PIL import Image
m = mujoco.MjModel.from_xml_path(sys.argv[1]); d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 900, 700)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.85]; cam.distance = 3.0; cam.azimuth = float(sys.argv[3]) if len(sys.argv) > 3 else 140; cam.elevation = -10
opt = mujoco.MjvOption(); opt.geomgroup[3] = int(os.environ.get('COL', 0))
r.update_scene(d, cam, opt); Image.fromarray(r.render()).save(sys.argv[2]); print('ok')
