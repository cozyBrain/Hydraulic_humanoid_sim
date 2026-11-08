import jax, jax.numpy as jnp, numpy as np, hydraulics as hy
from walk_env import HydroWalk
env = HydroWalk(); reset = jax.jit(env.reset); step = jax.jit(env.step)
for mode in ('zero', 'random'):
    s = reset(jax.random.PRNGKey(1)); acc = {}; R = 0
    for i in range(40):
        a = jnp.zeros(hy.NA) if mode == 'zero' else jax.random.uniform(jax.random.PRNGKey(i), (hy.NA,), minval=-1, maxval=1)
        s = step(s, a); R += float(s.reward)
        for k, v in s.metrics.items(): acc[k] = acc.get(k, 0) + float(v) / 40
        if float(s.done): break
    print(mode, 'steps', i + 1, 'return', round(R, 2), {k: round(v, 3) for k, v in acc.items()})
