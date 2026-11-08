"""Throughput check on your GPU for both training stages:  python bench.py [num_envs]"""
import sys, time, jax, jax.numpy as jnp
from walk_env import HydroWalk
n = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
print('devices:', jax.devices(), flush=True)
for mode, steps in (('fast', 1.5e8), ('full', 2e7)):
    env = HydroWalk(mode=mode)
    reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
    print(f'[{mode}] compiling ...', flush=True); t = time.time()
    s = reset(jax.random.split(jax.random.PRNGKey(0), n)); a = jnp.zeros((n, env.action_size))
    for _ in range(2): s = step(s, a)                       # 2 warm-up calls (the 2nd can still specialise types)
    jax.block_until_ready(s.reward); print(f'[{mode}] compiled in {time.time() - t:.0f} s, measuring ...', flush=True)
    t = time.time(); K = 20
    for _ in range(K): s = step(s, a)
    jax.block_until_ready(s.reward); el = time.time() - t; sps = n * K / el
    print(f'[{mode}] {n} envs: {sps:,.0f} control steps/s  ({sps * env.n_sub:,.0f} physics steps/s)  -> {steps:.1e} steps ~ {steps / sps / 3600:.1f} h (+~30% PPO)', flush=True)
