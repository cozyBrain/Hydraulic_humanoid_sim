"""Test bench: pelvis held in the air. Knee step 24 -> 100 deg, ankle pitch sine, hip pitch step. Tracking + pressures."""
import numpy as np, time, hydraulics as hy, sim_core as sc
S = sc.CpuSim(hung=True); J = hy.JOINTS; iK, iA, iH = J.index('knee_L'), J.index('ankle_pitch_L'), J.index('hip_pitch_L')
q0 = S.q0.copy(); t0 = time.time(); rows = []
for k in range(3000):
    t = k * 0.001; qt = q0.copy()
    if t > 0.3: qt[iK] = np.radians(100)
    if t > 1.5: qt[iK] = np.radians(10)
    qt[iA] = q0[iA] + np.radians(20) * np.sin(2 * np.pi * 1.5 * t)
    if t > 0.8: qt[iH] = np.radians(70)
    diag, u = S.step(qt)
    if k % 100 == 0:
        q = S.d.qpos[S.qadr]
        rows.append((t, np.degrees(qt[iK]), np.degrees(q[iK]), np.degrees(qt[iA]), np.degrees(q[iA]), np.degrees(qt[iH]), np.degrees(q[iH]),
                     S.h['pA'][6] / 1e5, S.h['pB'][6] / 1e5, float(diag['Ps']) / 1e5, float(diag['Pel']), float(S.h['w']) * 60 / 2 / np.pi))
print('%.0f steps/s' % (3000 / (time.time() - t0)))
print('  t   knee tgt/act   ankle tgt/act   hip tgt/act   knee pA/pB bar   Ps bar   Pel W   pump rpm')
for r in rows: print('%4.1f  %5.1f/%5.1f   %5.1f/%5.1f   %5.1f/%5.1f   %6.1f/%6.1f   %6.1f  %6.0f  %6.0f' % r)
