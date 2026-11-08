"""valves closed, every joint kicked: the oil-spring system must decay (numerical stability check at 1 ms)"""
import numpy as np, hydraulics as hy, sim_core as sc
S = sc.CpuSim(hung=True); S.d.qvel[S.dadr] = 0.5; peak = []
for k in range(600):
    S.step(S.q0, spool=np.zeros(hy.NA)); peak.append(np.abs(S.d.qvel[S.dadr]))
peak = np.array(peak)
for i, n in enumerate(hy.JOINTS):
    print('%-18s max|qd| 0-100ms %.2f  500-600ms %.3f' % (n, peak[:100, i].max(), peak[500:, i].max()))
