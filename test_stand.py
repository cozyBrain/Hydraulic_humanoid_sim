"""Smoke test: robot placed in a crouch, firmware holds the pose through the hydraulic servo loop for N seconds.
Logs pelvis height, supply pressure, pump speed, electrical power, a few chamber pressures."""
import sys, time, numpy as np, mujoco, jax, jax.numpy as jnp
from mujoco import mjx
import hydraulics as hy, sim_core as sc
T = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
m, qadr, dadr = sc.load()
P = {k: jnp.array(v) for k, v in hy.default_params().items()}
qpos0, q0 = sc.default_q(m, qadr)
d = mujoco.MjData(m); d.qpos[:] = qpos0; mujoco.mj_forward(m, d)
mx = mjx.put_model(m); dx = mjx.put_data(m, d)
h = hy.init_state(P, q0)
phys = sc.make_step(mx, qadr, dadr, P); servo = sc.make_servo(mx, qadr, dadr, P)
q_t = jnp.array(q0); KP, KD = jnp.array(sc.KP), jnp.array(sc.KD)
def one(carry, _):
    d, h = carry
    u, tau_des = servo(d, h, q_t, KP, KD)
    d, h, diag = phys(d, h, u, jnp.zeros(hy.NA))
    return (d, h), (d.qpos[2], diag['Ps'], h['w'], diag['Pel'], h['pA'][6], h['pB'][6], d.qpos[qadr[6]], h['T'], h['Voil'])
run = jax.jit(lambda d, h: jax.lax.scan(one, (d, h), None, length=100))
t0 = time.time(); (dx, h), log = run(dx, h); jax.block_until_ready(log); print('compile+0.1s', round(time.time() - t0, 1), 's')
logs = [log]; t0 = time.time()
for i in range(int(T / 0.1) - 1):
    (dx, h), log = run(dx, h); logs.append(log)
jax.block_until_ready(log); el = time.time() - t0
print('sim %.1fs in %.1fs wall (%.0f steps/s, 1 env CPU)' % (T, el, (T - 0.1) * 1000 / el))
L = [np.concatenate([np.array(l[i]) for l in logs]) for i in range(9)]
for t in np.linspace(0, len(L[0]) - 1, 11).astype(int):
    print(f't={t/1000:5.2f}s  z={L[0][t]:.3f} m  Ps={L[1][t]/1e5:6.1f} bar  pump={L[2][t]*60/2/np.pi:6.0f} rpm  Pel={L[3][t]:7.0f} W  '
          f'knee_L pA/pB={L[4][t]/1e5:5.1f}/{L[5][t]/1e5:5.1f} bar  knee={np.degrees(L[6][t]):5.1f} deg  oil {L[7][t]:.1f}C  acc {L[8][t]*1e3:.0f} mL')
