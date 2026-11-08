"""headless check of the viewer's MANUAL + gantry path: slider (d.ctrl, degrees) -> firmware -> hydraulics"""
import numpy as np, mujoco, hydraulics as hy, sim_core as sc
S = sc.CpuSim(f'{sc.HERE}/model/hydro_humanoid_viz.xml'); m, d = S.m, S.d
pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in hy.JOINTS])
d.ctrl[act] = np.degrees(S.q0); gp = d.xpos[pel].copy() + [0, 0, 0.15]; gq = np.array([1.0, 0, 0, 0])
def gantry():
    F = 30000 * (gp - d.xpos[pel]) - 3000 * d.qvel[0:3]
    qe = np.zeros(4); qi = np.zeros(4); mujoco.mju_negQuat(qi, d.xquat[pel]); mujoco.mju_mulQuat(qe, gq, qi)
    rv = np.zeros(3); mujoco.mju_quat2Vel(rv, qe, 1.0); return F, 4000 * rv - 400 * d.qvel[3:6]
J = hy.JOINTS
for k in range(3000):
    if k == 500: d.ctrl[act[J.index('knee_L')]] = 110; d.ctrl[act[J.index('hip_pitch_L')]] = 80
    if k == 1500: d.ctrl[act[J.index('shoulder_pitch_R')]] = 120; d.ctrl[act[J.index('elbow_R')]] = 90; d.ctrl[act[J.index('waist_yaw')]] = 25
    d.xfrc_applied[:] = 0; F, T = gantry(); d.xfrc_applied[pel, :3] = F; d.xfrc_applied[pel, 3:] = T
    diag, u = S.step(np.radians(d.ctrl[act]))
    if k % 500 == 499:
        q = np.degrees(d.qpos[S.qadr])
        print('t %.1f pelvis z %.3f  knee_L %.0f hip_L %.0f  sh_R %.0f elbow_R %.0f waist_yaw %.0f  Pel %.0f W' % (
            k / 1000, d.xpos[pel][2], q[J.index('knee_L')], q[J.index('hip_pitch_L')], q[J.index('shoulder_pitch_R')], q[J.index('elbow_R')], q[J.index('waist_yaw')], diag['Pel']))
