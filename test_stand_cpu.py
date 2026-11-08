import numpy as np, mujoco, sim_core as sc, hydraulics as hy, sys
S = sc.CpuSim(); d = S.d; J = hy.JOINTS
S.sole = [mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_GEOM, f'sole_{s}') for s in 'LR']
bal = len(sys.argv) > 1
for k in range(5000):
    qt = S.q0.copy()
    if bal:   # demo balance: COM over the feet (ankle strategy) + keep pelvis upright (hip strategy)
        w, x, y, z = d.qpos[3:7]; pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)); roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        com = d.subtree_com[1]; fx = 0.5 * (d.geom_xpos[S.sole[0]] + d.geom_xpos[S.sole[1]]); vcom = d.qvel[0:3]
        for s in 'LR':
            qt[J.index(f'ankle_pitch_{s}')] += 4.0 * (com[0] - fx[0]) + 0.6 * vcom[0]
            qt[J.index(f'ankle_roll_{s}')] += -3.0 * (com[1] - fx[1]) - 0.4 * vcom[1]
            qt[J.index(f'hip_pitch_{s}')] += -0.8 * pitch - 0.05 * d.qvel[4]
            qt[J.index(f'hip_roll_{s}')] += 0.8 * roll + 0.05 * d.qvel[3]
    diag, u = S.step(qt)
    if k % 500 == 0: print('t %.1f z %.3f x %.3f pitch %.1f deg  Ps %.0f bar  Pel %.0f W' % (k / 1000, d.qpos[2], d.qpos[0], np.degrees(2 * np.arcsin(d.qpos[5])), float(diag['Ps']) / 1e5, float(diag['Pel'])))
