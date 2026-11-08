import time, jax, jax.numpy as jnp, numpy as np, hydraulics as hy
from walk_env import HydroWalk
env = HydroWalk()
reset = jax.jit(env.reset); step = jax.jit(env.step)
t = time.time(); s = reset(jax.random.PRNGKey(0)); jax.block_until_ready(s.obs); print('reset compile', round(time.time() - t, 1))
t = time.time(); s = step(s, jnp.zeros(hy.NA)); jax.block_until_ready(s.obs); print('step compile', round(time.time() - t, 1))
t = time.time(); rs = []
for i in range(50):
    s = step(s, jnp.zeros(hy.NA)); rs.append((float(s.reward), float(s.done), float(s.pipeline_state.qpos[2]), float(s.metrics['power_W']), float(s.metrics['Ps_bar'])))
jax.block_until_ready(s.obs); el = time.time() - t
print('50 ctrl steps (1 s sim) in %.1fs' % el)
for i in range(0, 50, 7): print(i, np.round(rs[i], 3))
print({k: v.shape for k, v in s.obs.items()})
