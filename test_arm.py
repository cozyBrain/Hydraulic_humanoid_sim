"""hung bench for the arm: shoulder pitch raise, elbow flex, wrist moves"""
import numpy as np, hydraulics as hy, sim_core as sc
S = sc.CpuSim(hung=True); J = hy.JOINTS
idx = {n: J.index(n) for n in ('shoulder_pitch_L', 'shoulder_roll_L', 'elbow_L', 'wrist_pitch_L', 'shoulder_yaw_L')}
for k in range(2500):
    t = k / 1000; qt = S.q0.copy()
    if t > 0.2: qt[idx['shoulder_pitch_L']] = np.radians(90)
    if t > 0.8: qt[idx['elbow_L']] = np.radians(100)
    if t > 1.4: qt[idx['shoulder_roll_L']] = np.radians(60); qt[idx['wrist_pitch_L']] = np.radians(40)
    if t > 1.9: qt[idx['shoulder_yaw_L']] = np.radians(-45)
    diag, u = S.step(qt)
    if k % 250 == 0:
        q = S.d.qpos[S.qadr]
        print('t %.2f ' % t + '  '.join('%s %.0f/%.0f' % (n.replace('_L', ''), np.degrees(qt[i]), np.degrees(q[i])) for n, i in idx.items()) + '  Pel %.0f W' % diag['Pel'])
