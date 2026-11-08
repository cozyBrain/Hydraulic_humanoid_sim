import time, sys, jax, jax.numpy as jnp, numpy as np, hydraulics as hy
from walk_env import HydroWalk
for mode in ('fast', 'full'):
    env = HydroWalk(mode=mode); reset = jax.jit(env.reset); step = jax.jit(env.step)
    s = reset(jax.random.PRNGKey(1)); s = step(s, jnp.zeros(hy.NA)); jax.block_until_ready(s.reward)
    t = time.time(); R = 0
    for i in range(50): s = step(s, jnp.zeros(hy.NA)); R += float(s.reward)
    print(mode, 'n_sub', env.n_sub, '50 steps %.2fs' % (time.time() - t), 'return', round(R, 1), 'z', round(float(s.pipeline_state.qpos[2]), 3), 'power', round(float(s.metrics['power_W'])), 'Ps', round(float(s.metrics['Ps_bar']), 1))
