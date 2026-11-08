"""Walking-policy training (PPO on MJX) with a live terminal dashboard, auto-resume and a live 3D view.

    python train_walk.py                          # trains runs/walk: stage 1 (fast actuator model) then stage 2 (full hydraulics)
    python train_walk.py --envs 512               # fewer parallel robots (less GPU memory)
    python watch.py runs/walk                     # in a second terminal: watch training robots live in 3D

Ctrl+C at any time: a checkpoint is written and the next start continues exactly from there (optimizer state included).
Files in the run folder: ckpt.pkl (resume), policy.pkl (for viewer.py / rollout_video.py), log.csv, live.npz (for watch.py)."""
import argparse, os, sys, time, pickle, signal, threading, json, csv
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--run', default='runs/walk')
ap.add_argument('--envs', type=int, default=1024)
ap.add_argument('--fast-steps', type=float, default=1.5e8, help='stage 1 control steps (fast actuator model, 4 ms physics)')
ap.add_argument('--full-steps', type=float, default=2e7, help='stage 2 control steps (full hydraulics, 1 ms physics)')
ap.add_argument('--unroll', type=int, default=20)
ap.add_argument('--minibatches', type=int, default=4)
ap.add_argument('--epochs', type=int, default=4)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--entropy', type=float, default=5e-3)
ap.add_argument('--gamma', type=float, default=0.99)
ap.add_argument('--episode', type=int, default=1000, help='episode length in control steps (50 Hz)')
ap.add_argument('--ckpt-min', type=float, default=5.0, help='checkpoint every N minutes')
ap.add_argument('--live-envs', type=int, default=4, help='robots streamed to watch.py')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--init', default=None, help='start a NEW run from the weights of another ckpt.pkl (warm start)')
ap.add_argument('--start-stage', default='fast', choices=['fast', 'full'], help='with --init: which stage to start in')
a = ap.parse_args()
os.makedirs(a.run, exist_ok=True)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.text import Text
from rich.console import Group
con = Console()
con.print(Panel.fit(f'[bold]Hydraulic humanoid v9 — walking policy training[/]\nrun folder: [cyan]{a.run}[/]   parallel robots: {a.envs}\n'
                    f'live 3D view: [green]python watch.py {a.run}[/] (other terminal)   stop: [yellow]Ctrl+C[/] (checkpoint is saved, rerun to resume)'))

with con.status('[bold]starting JAX / CUDA ...') as st:
    import jax, jax.numpy as jnp, optax
    cache = os.path.join(os.path.dirname(os.path.abspath(a.run)), '.jax_cache')
    jax.config.update('jax_compilation_cache_dir', cache); jax.config.update('jax_persistent_cache_min_compile_time_secs', 2.0)
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics, specs
    import hydraulics as hy
    from walk_env import HydroWalk, domain_randomize
con.print(f'devices: {jax.devices()}   (compiled code is cached in {cache} -> restarts compile much faster)')

CKPT, POL, LOG, LIVE = [os.path.join(a.run, f) for f in ('ckpt.pkl', 'policy.pkl', 'log.csv', 'live.npz')]
STAGES = [('fast', a.fast_steps), ('full', a.full_steps)]
ck = pickle.load(open(CKPT, 'rb')) if os.path.exists(CKPT) else None
if ck is None and a.init:
    ck = pickle.load(open(a.init, 'rb')); ck = dict(stage=a.start_stage, env_steps=0, iteration=0, params=ck['params'], opt_state=None, key=ck['key'], warm=True)
    con.print(f'[green]warm start[/] from {a.init} -> stage {a.start_stage}')
elif ck: con.print(f"[green]resuming[/] from {CKPT}: stage {ck['stage']}, {ck['env_steps']:,.0f} steps done, iteration {ck['iteration']}")

def to_np(t): return jax.tree.map(lambda x: np.asarray(x), t)

def run_stage(stage, target, ck):
    env = HydroWalk(mode=stage)
    obs_size = env.observation_size; nA = env.action_size; N = a.envs; T = a.unroll
    net = ppo_networks.make_ppo_networks(obs_size, nA, preprocess_observations_fn=running_statistics.normalize,
                                         policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 256, 128),
                                         policy_obs_key='state', value_obs_key='privileged_state', init_noise_std=0.6)
    dist = net.parametric_action_distribution
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    key = jax.random.PRNGKey(a.seed)
    if ck is not None:
        params = jax.tree.map(jnp.asarray, ck['params']); key = jnp.asarray(ck['key'])
        opt_state = jax.tree.map(jnp.asarray, ck['opt_state']) if ck.get('opt_state') is not None else None
        if ck['stage'] != stage or ck.get('opt_state') is None: opt_state = opt.init(params[1:])                           # fresh optimizer moments for the new stage
        steps0 = ck['env_steps'] if ck['stage'] == stage else 0; it0 = ck['iteration'] if ck['stage'] == stage else 0
    else:
        k1, k2, key = jax.random.split(key, 3)
        norm = running_statistics.init_state({k: specs.Array(v, jnp.dtype('float32')) for k, v in obs_size.items()})
        params = (norm, net.policy_network.init(k1), net.value_network.init(k2)); opt_state = opt.init(params[1:]); steps0 = 0; it0 = 0
    # ---- per-robot randomised physics (friction, masses, damping) + vectorised env
    key, kdr = jax.random.split(key)
    sys_v, axes = domain_randomize(env.sys, jax.random.split(kdr, N))
    def _reset(sys, k): env.sys = sys; return env.reset(k)
    def _step(sys, s, act): env.sys = sys; return env.step(s, act)
    v_reset = jax.vmap(_reset, in_axes=(axes, 0)); v_step = jax.vmap(_step, in_axes=(axes, 0, 0))
    def _reobs(sys, s, info, k): env.sys = sys; return env._obs(s.pipeline_state, info['h'], info, k)
    v_reobs = jax.vmap(_reobs, in_axes=(axes, 0, 0, 0))

    def policy_logits(p, obs): return net.policy_network.apply(p[0], p[1], obs)
    def value(p, obs): return net.value_network.apply(p[0], p[2], obs)

    def rollout(params, carry, key):
        st, first, ep_ret, ep_len = carry
        def one(c, k):
            st, first, ep_ret, ep_len, stats = c
            k1, k2, k3 = jax.random.split(k, 3)
            logits = policy_logits(params, st.obs)
            raw = dist.sample_no_postprocessing(logits, k1); logp = dist.log_prob(logits, raw); act = dist.postprocess(raw)
            nst = v_step(sys_v, st, act)
            fell = nst.done > 0; trunc = (nst.info['step'] >= a.episode) & ~fell; reset = fell | trunc
            ep_ret = ep_ret + nst.reward; ep_len = ep_len + 1
            stats = dict(n=stats['n'] + reset.sum(), ret=stats['ret'] + jnp.where(reset, ep_ret, 0).sum(), len=stats['len'] + jnp.where(reset, ep_len, 0).sum(),
                         falls=stats['falls'] + fell.sum(), vel=stats['vel'] + nst.metrics['r_vel'].sum(), imit=stats['imit'] + nst.metrics['r_imit'].sum(), turn=stats['turn'] + nst.metrics['p_turn'].sum(), power=stats['power'] + nst.metrics['power_W'].sum(),
                         Ps=stats['Ps'] + nst.metrics['Ps_bar'].sum())
            term_obs = nst.obs
            # auto-reset: back to this robot's start state, fresh command / hydraulic randomisation / rng
            info_r = dict(first.info); kk = jax.random.split(k2, N)
            info_r['rng'] = jax.vmap(lambda q: jax.random.fold_in(q, 1))(kk)
            info_r['cmd'] = jax.vmap(env._sample_cmd)(kk)
            info_r['hscale'] = 1.0 + jax.vmap(lambda q: jax.random.uniform(q, (5,), minval=-0.15, maxval=0.15))(jax.vmap(lambda q: jax.random.fold_in(q, 2))(kk))
            obs_r = v_reobs(sys_v, first, info_r, jax.random.split(k3, N))
            sel = lambda x, y: jnp.where(reset.reshape((-1,) + (1,) * (x.ndim - 1)), x, y)
            nst = nst.replace(pipeline_state=jax.tree.map(sel, first.pipeline_state, nst.pipeline_state), obs=jax.tree.map(sel, obs_r, nst.obs),
                              info=jax.tree.map(sel, info_r, nst.info))
            ep_ret = jnp.where(reset, 0.0, ep_ret); ep_len = jnp.where(reset, 0, ep_len)
            tr = dict(obs=st.obs, raw=raw, logp=logp, rew=nst.reward,
                      done=fell.astype(jnp.float32), trunc=trunc.astype(jnp.float32), term_obs=term_obs,
                      live_q=nst.pipeline_state.qpos[:a.live_envs], live_x=jnp.stack([nst.info['Ps'][:a.live_envs] / 1e5, nst.info['h']['w'][:a.live_envs],
                                                                                       nst.metrics['power_W'][:a.live_envs]], -1))
            return (nst, first, ep_ret, ep_len, stats), tr
        z = jnp.zeros((), jnp.float32)
        stats0 = dict(n=jnp.zeros((), jnp.int32), ret=z, len=z, falls=jnp.zeros((), jnp.int32), vel=z, imit=z, turn=z, power=z, Ps=z)
        (st, first, ep_ret, ep_len, stats), traj = jax.lax.scan(one, (st, first, ep_ret, ep_len, stats0), jax.random.split(key, T))
        return (st, first, ep_ret, ep_len), traj, stats

    def gae(params, traj, last_obs):
        v = value(params, traj['obs']); v_last = value(params, last_obs); v_term = value(params, traj['term_obs'])
        rew = traj['rew'] + a.gamma * v_term * traj['trunc']                # bootstrap through time-limit truncation
        cont = 1.0 - jnp.maximum(traj['done'], traj['trunc'])
        v_next = jnp.concatenate([v[1:], v_last[None]], 0)
        delta = rew + a.gamma * v_next * cont - v
        def back(c, x): d, cn = x; c = d + a.gamma * 0.95 * cn * c; return c, c
        _, adv = jax.lax.scan(back, jnp.zeros_like(v_last), (delta, cont), reverse=True)
        return adv, adv + v

    def loss_fn(pv, norm, mb, k):
        p = (norm,) + pv
        logits = policy_logits(p, mb['obs']); logp = dist.log_prob(logits, mb['raw'])
        ratio = jnp.exp(logp - mb['logp']); adv = (mb['adv'] - mb['adv'].mean()) / (mb['adv'].std() + 1e-8)
        pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv))
        v = value(p, mb['obs']); vl = 0.5 * jnp.mean((mb['ret'] - v) ** 2)
        ent = jnp.mean(dist.entropy(logits, k))
        return pg + 0.5 * vl - a.entropy * ent, (pg, vl, ent)

    @jax.jit
    def iteration(params, opt_state, carry, key):
        k1, k2 = jax.random.split(key)
        carry, traj, stats = rollout(params, carry, k1)
        norm = running_statistics.update(params[0], traj['obs'])
        params = (norm,) + params[1:]
        adv, ret = gae(params, traj, carry[0].obs)
        data = dict(obs=traj['obs'], raw=traj['raw'], logp=traj['logp'], adv=adv, ret=ret)
        data = jax.tree.map(lambda x: x.reshape((T * N,) + x.shape[2:]), data)
        def epoch(c, k):
            pv, os_ = c; ka, kb = jax.random.split(k); perm = jax.random.permutation(ka, T * N)
            mbs = jax.tree.map(lambda x: x[perm].reshape((a.minibatches, -1) + x.shape[1:]), data)
            def mb_step(c2, x):
                pv, os_ = c2; mb, kk = x
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(pv, norm, mb, kk)
                upd, os_ = opt.update(g, os_, pv); pv = optax.apply_updates(pv, upd); return (pv, os_), aux
            (pv, os_), aux = jax.lax.scan(mb_step, (pv, os_), (mbs, jax.random.split(kb, a.minibatches)))
            return (pv, os_), aux
        (pv, opt_state), aux = jax.lax.scan(epoch, (params[1:], opt_state), jax.random.split(k2, a.epochs))
        params = (norm,) + pv
        logits = policy_logits(params, carry[0].obs); std = jnp.mean(jax.nn.softplus(logits[..., nA:]) + 0.001)
        m = dict(pg=aux[0].mean(), vl=aux[1].mean(), ent=aux[2].mean(), std=std, rew=traj['rew'].mean(), **stats)
        return params, opt_state, carry, m, traj['live_q'], traj['live_x']

    # ---- start robots
    key, kr = jax.random.split(key)
    first = jax.jit(v_reset)(sys_v, jax.random.split(kr, N))
    carry = (first, first, jnp.zeros(N), jnp.zeros(N, jnp.int32))
    iters_needed = int(np.ceil((target - steps0) / (N * T)))
    if iters_needed <= 0: return params, opt_state, key, steps0, it0
    # ---- dashboard
    hist = dict(n=0, ret=[], len=[], falls=0, vel=[], imit=[], turn=[], power=[], Ps=[])
    live_q, live_x, live_meta = [], [], []
    state = dict(stage=stage, steps=steps0, it=it0, t_ck=time.time(), last_ck='-', ips=None, t_last=None, compiling=True, t0=time.time(), m=None, stop=False)
    total_all = sum(t for _, t in STAGES)
    done_before = sum(t for s_, t in STAGES[:[s for s, _ in STAGES].index(stage)])
    def save(params, opt_state, key, final=False):
        ckd = dict(stage=stage, env_steps=state['steps'], iteration=state['it'], params=to_np(params), opt_state=to_np(opt_state), key=np.asarray(key), args=vars(a))
        pickle.dump(ckd, open(CKPT + '.tmp', 'wb')); os.replace(CKPT + '.tmp', CKPT)
        pickle.dump(to_np(params), open(POL + '.tmp', 'wb')); os.replace(POL + '.tmp', POL)
        state['t_ck'] = time.time(); state['last_ck'] = time.strftime('%H:%M:%S')
    def render():
        tb = Table.grid(padding=(0, 2)); tb.add_column(style='bold'); tb.add_column(); tb.add_column(style='bold'); tb.add_column()
        el = time.time() - state['t0']
        if state['compiling']:
            return Panel(Group(Text(f'stage {stage}: compiling the simulator + PPO step for the GPU ...  {el:5.0f} s'),
                               Text('first run on a GTX 1060 can take several minutes; it is cached, later starts are much faster', style='dim')),
                         title='training', border_style='yellow')
        sps = state['ips'] * N * T if state['ips'] else 0
        rem_stage = max(target - state['steps'], 0); eta_stage = rem_stage / sps if sps else 0
        rem_all = rem_stage + sum(t for s_, t in STAGES[[s for s, _ in STAGES].index(stage) + 1:])
        eta_all = rem_all / sps if sps else 0      # (stage 2 is slower per step; refined when it starts)
        fmt = lambda s: f'{int(s // 3600)}h {int(s % 3600 // 60):02d}m {int(s % 60):02d}s'
        m = state['m'] or {}
        r = lambda k: (np.mean(hist[k][-50:]) if hist[k] else float('nan'))
        tb.add_row('stage', f"{stage} ({'reduced model, 4 ms' if stage == 'fast' else 'full hydraulics, 1 ms'})", 'iteration', f"{state['it']:,}")
        tb.add_row('steps', f"{state['steps']:,.0f} / {target:,.0f}", 'speed', f'{sps:,.0f} steps/s   ({1 / state["ips"]:.2f} s/iter)' if state['ips'] else '-')
        tb.add_row('ETA stage', fmt(eta_stage), 'ETA total', fmt(eta_all) + ' (approx.)')
        tb.add_row('episodes', f"{hist['n']:,}", 'mean return', f"{r('ret'):.1f}")
        tb.add_row('episode length', f"{r('len') * 0.02:.1f} s  (max {a.episode * 0.02:.0f} s)", 'falls', f"{100 * hist['falls'] / max(hist['n'], 1):.0f} % of episodes")
        tb.add_row('mocap match', f"{r('imit'):.2f}  (1 = human gait, while walking forward)", 'turn error', f"{r('turn'):.2f} rad/s  (0 = turns exactly as told)")
        tb.add_row('vel tracking', f"{r('vel'):.2f}  (1 = perfect)", 'hydraulic power', f"{r('power'):.0f} W   Ps {r('Ps'):.0f} bar")
        tb.add_row('policy std', f"{float(m.get('std', np.nan)):.3f}", 'losses', f"pg {float(m.get('pg', np.nan)):+.3f}  v {float(m.get('vl', np.nan)):.3f}  ent {float(m.get('ent', np.nan)):.2f}")
        tb.add_row('checkpoint', f"{state['last_ck']}  (every {a.ckpt_min:g} min + on Ctrl+C)", 'elapsed', fmt(el))
        frac = (done_before + state['steps']) / total_all
        bars = Table.grid(padding=(0, 1)); bars.add_column(style='bold', width=16); bars.add_column(width=52); bars.add_column()
        bars.add_row(f'stage {stage}', ProgressBar(total=target, completed=state['steps'], width=50), f"{100 * state['steps'] / target:5.1f} %")
        bars.add_row('overall', ProgressBar(total=total_all, completed=done_before + state['steps'], width=50), f'{100 * frac:5.1f} %')
        return Panel(Group(tb, Text(''), bars), title=f'training — {a.run}', border_style='green')
    def on_sigint(sig, frm):
        if state['stop']: raise KeyboardInterrupt          # second Ctrl+C: hard stop
        state['stop'] = True
    signal.signal(signal.SIGINT, on_sigint)
    logf = open(LOG, 'a', newline=''); lw = csv.writer(logf)
    if os.path.getsize(LOG) == 0: lw.writerow(['time', 'stage', 'iteration', 'env_steps', 'mean_return', 'mean_len_s', 'fall_rate', 'vel_track', 'power_W', 'Ps_bar', 'std', 'pg', 'vloss'])
    with Live(render(), console=con, refresh_per_second=4) as live:
        stop_spin = threading.Event()
        def spin():
            while not stop_spin.is_set(): live.update(render()); time.sleep(0.5)
        th = threading.Thread(target=spin, daemon=True); th.start()
        for i in range(iters_needed):
            key, k = jax.random.split(key)
            params, opt_state, carry, m, lq, lx = iteration(params, opt_state, carry, k)
            m = jax.device_get(m); t = time.time()
            if state['compiling']: state['compiling'] = False; state['t0'] = t
            elif state['t_last']: dt_ = t - state['t_last']; state['ips'] = 1 / dt_ if state['ips'] is None else 0.9 * state['ips'] + 0.1 / dt_
            state['t_last'] = t; state['it'] += 1; state['steps'] += N * T; state['m'] = m
            if int(m['n']) > 0:
                n = int(m['n']); hist['n'] += n; hist['falls'] += int(m['falls'])
                hist['ret'].append(float(m['ret']) / n); hist['len'].append(float(m['len']) / n)
            hist['vel'].append(float(m['vel']) / (N * T)); hist['imit'].append(float(m['imit']) / (N * T)); hist['turn'].append(float(m['turn']) / (N * T)); hist['power'].append(float(m['power']) / (N * T)); hist['Ps'].append(float(m['Ps']) / (N * T))
            # live stream for watch.py (a few robots, cheap)
            lq = np.asarray(lq); lx = np.asarray(lx); live_q.append(lq); live_x.append(lx)
            live_meta.append(np.array([[state['it'], state['steps'], hist['ret'][-1] if hist['ret'] else np.nan]] * T))
            if len(live_q) > 150: live_q.pop(0); live_x.pop(0); live_meta.pop(0)
            if state['it'] % 2 == 0:
                np.savez(LIVE + '.tmp.npz', q=np.concatenate(live_q), x=np.concatenate(live_x), meta=np.concatenate(live_meta), stage=stage,
                         frame_end=state['it'] * T, dt=0.02)
                os.replace(LIVE + '.tmp.npz', LIVE)
            if state['it'] % 10 == 0:
                lw.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), stage, state['it'], state['steps'], hist['ret'][-1] if hist['ret'] else '', hist['len'][-1] * 0.02 if hist['len'] else '',
                             hist['falls'] / max(hist['n'], 1), hist['vel'][-1], hist['power'][-1], hist['Ps'][-1], float(m['std']), float(m['pg']), float(m['vl'])]); logf.flush()
            if time.time() - state['t_ck'] > a.ckpt_min * 60: save(params, opt_state, key)
            if state['stop']: break
        save(params, opt_state, key); stop_spin.set(); live.update(render())
    if state['stop']:
        con.print(f'[yellow]stopped by user[/] — checkpoint saved ({state["steps"]:,.0f} steps). Run the same command again to continue.'); sys.exit(0)
    return params, opt_state, key, state['steps'], state['it']

stages = [s for s, _ in STAGES]; start = stages.index(ck['stage']) if ck else 0
for si in range(start, len(STAGES)):
    stage, target = STAGES[si]
    if ck and ck['stage'] == stage and ck['env_steps'] >= target: continue
    params, opt_state, key, steps, it = run_stage(stage, target, ck)
    ck = dict(stage=stage, env_steps=steps, iteration=it, params=to_np(params), opt_state=to_np(opt_state), key=np.asarray(key))
    con.print(f'[green]stage {stage} finished[/] ({steps:,.0f} steps)')
con.print(f'[bold green]training complete.[/] policy: {POL}   ->  python viewer.py {POL}')
