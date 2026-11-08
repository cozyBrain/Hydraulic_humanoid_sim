"""Look inside the trained neural network.
usage:  python inspect_policy.py runs/walk/policy.pkl        -> prints the structure + writes policy_report.png
Shows: layer sizes / parameter counts, weight distributions per layer, which sensor inputs the policy relies on most
(first-layer weight magnitude x input spread), and the learned exploration noise per joint."""
import os; os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # tools run on CPU: the GPU belongs to the trainer
import sys, pickle, numpy as np, jax
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import hydraulics as hy
pk = sys.argv[1] if len(sys.argv) > 1 else 'runs/walk/policy.pkl'
norm, pol, val = pickle.load(open(pk, 'rb'))
J = hy.JOINTS; nA = hy.NA
names = (['gravity x', 'gravity y', 'gravity z', 'gyro x', 'gyro y', 'gyro z'] + [f'q {j}' for j in J] + [f'qd {j}' for j in J] + [f'torque {j}' for j in J]
         + [f'last action {j}' for j in J] + ['cmd vx', 'cmd vy', 'cmd yaw', 'phase sin', 'phase cos', 'supply pressure'])
groups = {'IMU (gravity, gyro)': range(0, 6), 'joint angles': range(6, 6 + nA), 'joint velocities': range(6 + nA, 6 + 2 * nA),
          'joint torques (from oil pressure)': range(6 + 2 * nA, 6 + 3 * nA), 'previous action': range(6 + 3 * nA, 6 + 4 * nA),
          'command + gait phase': range(6 + 4 * nA, 11 + 4 * nA), 'supply pressure': range(11 + 4 * nA, 12 + 4 * nA)}
def layers(p): return [(k, np.asarray(v['kernel']), np.asarray(v['bias'])) for k, v in sorted(p['params'].items())]
print(f'\nfile: {pk}')
for title, p in (('POLICY network (actor): sensors -> joint targets', pol), ('VALUE network (critic, training only)', val)):
    L = layers(p); n = sum(k.size + b.size for _, k, b in L)
    print(f'\n{title}:  {n:,} parameters')
    for name, k, b in L: print(f'  {name:9s} {k.shape[0]:4d} -> {k.shape[1]:4d}   weights mean|w| {np.abs(k).mean():.4f}   max|w| {np.abs(k).max():.3f}')
W0 = layers(pol)[0][1]
imp = np.abs(W0).sum(1)                                                # inputs are normalised before layer 0 -> |W| is comparable
print('\nwhat the policy looks at most (first-layer weight, top 15):')
for i in np.argsort(-imp)[:15]: print(f'  {names[i]:28s} {imp[i]:.2f}')
print('\nby sensor group (mean per input):')
for g, r in groups.items(): print(f'  {g:34s} {imp[list(r)].mean():.2f}')
print(f"\nobservation normaliser has seen {float(np.asarray(norm.count)):,.0f} samples")

fig, ax = plt.subplots(2, 2, figsize=(14, 9))
for name, k, b in layers(pol): ax[0, 0].hist(k.ravel(), bins=80, histtype='step', label=f'{name} {k.shape[0]}x{k.shape[1]}')
ax[0, 0].set_title('policy weight distributions'); ax[0, 0].legend(fontsize=8)
gi = [imp[list(r)].mean() for r in groups.values()]
ax[0, 1].barh(list(groups.keys()), gi, color='#2a78d6'); ax[0, 1].set_title('which sensors the policy uses (first-layer |w|)')
ax[1, 0].imshow(np.abs(W0).T, aspect='auto', cmap='magma'); ax[1, 0].set_title('first layer |weights|  (x: input, y: neuron)')
ax[1, 0].set_xlabel('input index (IMU | angles | velocities | torques | last action | cmd)')
lastk = layers(pol)[-1][1]; ax[1, 1].barh(J, np.abs(lastk[:, :nA]).sum(0), color='#eb6834'); ax[1, 1].set_title('output layer |w| per joint (how actively each joint is driven)')
ax[1, 1].tick_params(axis='y', labelsize=7)
plt.tight_layout(); out = pk.replace('.pkl', '_report.png'); plt.savefig(out, dpi=110); print(f'\nfigure: {out}')
