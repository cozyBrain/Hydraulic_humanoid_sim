import jax, jax.numpy as jnp, numpy as np, hydraulics as hy
from walk_env import HydroWalk, domain_randomize
import sys
env = HydroWalk(mode=sys.argv[1] if len(sys.argv) > 1 else 'fast'); N = 8
sys_v, axes = domain_randomize(env.sys, jax.random.split(jax.random.PRNGKey(0), N))
def _reset(sys, k): env.sys = sys; return env.reset(k)
def _step(sys, s, a): env.sys = sys; return env.step(s, a)
r = jax.jit(jax.vmap(_reset, in_axes=(axes, 0))); st = jax.jit(jax.vmap(_step, in_axes=(axes, 0, 0)))
s = r(sys_v, jax.random.split(jax.random.PRNGKey(1), N))
for i in range(int(sys.argv[2]) if len(sys.argv) > 2 else 30):
    a = jax.random.uniform(jax.random.PRNGKey(i), (N, hy.NA), minval=-1, maxval=1)
    s = st(sys_v, s, a)
    bad = {k: bool(jnp.any(~jnp.isfinite(v))) for k, v in s.metrics.items()}
    if any(bad.values()) or not bool(jnp.all(jnp.isfinite(s.reward))):
        print('step', i, bad, 'reward', np.asarray(s.reward), 'qpos finite', bool(jnp.all(jnp.isfinite(s.pipeline_state.qpos))),
              'h', {k: bool(jnp.all(jnp.isfinite(v))) for k, v in s.info['h'].items()}); break
else: print('no NaN in 30 steps', np.asarray(s.metrics['power_W']))
