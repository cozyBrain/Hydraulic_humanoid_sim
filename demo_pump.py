"""Pump close-up demo: the robot squats (balance controller + squat targets) so the actuators draw oil;
watch the BLDC speed up the pinion / ring gear behind the glass.  The WHOLE simulation is recorded in slow motion
(default 1/33: 0.5 s of simulated time -> 17 s of video; at 4000 rpm the gears need that much slow-down) so the gear teeth can be followed by eye.
usage: python demo_pump.py out.mp4 [sim_seconds] [slowdown]"""
import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # tools run on CPU: the GPU belongs to the trainer
import sys, numpy as np, mujoco, imageio
from PIL import Image, ImageDraw
import hydraulics as hy, sim_core as sc, viz_util as vu
out = sys.argv[1] if len(sys.argv) > 1 else 'pump.mp4'; T = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5; slow = float(sys.argv[3]) if len(sys.argv) > 3 else 33
S = sc.CpuSim(f'{sc.HERE}/model/hydro_humanoid_viz.xml'); m, d = S.m, S.d; poser = vu.LinkagePoser(m); J = hy.JOINTS
r = mujoco.Renderer(m, 720, 960); cam = mujoco.MjvCamera(); cam.distance = 0.75; cam.elevation = -8; cam.azimuth = 200
pump = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'mc_pumpshaft')
frames, ang = [], 0.0
for k in range(int(T * 1000)):
    t = k / 1000; qt = sc.stand_targets(S)
    s = 1.0 if t > 0.08 else 0.0                                   # sudden move at t = 0.08 s: arms up fast + dip
    for sd in 'LR':
        qt[J.index(f'knee_{sd}')] += np.radians(25) * s; qt[J.index(f'hip_pitch_{sd}')] += np.radians(12) * s; qt[J.index(f'ankle_pitch_{sd}')] -= np.radians(12) * s
        qt[J.index(f'shoulder_pitch_{sd}')] += np.radians(90) * s; qt[J.index(f'elbow_{sd}')] += np.radians(60) * s
    diag, u = S.step(qt); ang += float(S.h['w']) * 0.001
    if k % max(1, int(round(1000 / 30 / slow))) == 0:
        poser(d, ang); mujoco.mj_forward(m, d); cam.lookat[:] = d.xpos[pump]
        r.update_scene(d, cam); img = Image.fromarray(r.render()); dr = ImageDraw.Draw(img)
        dr.text((16, 14), 't = %.3f s   pump %5.0f rpm   Ps %4.0f bar   P_el %4.1f kW   acc %3.0f mL   (slow motion 1/%d)' %
                (t, float(S.h['w']) * 9.549, float(diag['Ps']) / 1e5, float(diag['Pel']) / 1e3, float(S.h['Voil']) * 1e6, slow), fill=(235, 235, 230))
        frames.append(np.array(img))
imageio.mimsave(out, frames, fps=30, quality=8); print('wrote', out, len(frames), 'frames')
