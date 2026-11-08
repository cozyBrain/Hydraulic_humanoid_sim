"""PPO training of the walking policy (Brax PPO on MJX).  GPU strongly recommended.
usage:  python train.py                    # full run (defaults tuned for one RTX 3080-4090 class GPU)
        python train.py --smoke            # tiny CPU run to check everything is wired
        python train.py --steps 2e8 --envs 4096 --out runs/walk2"""
import argparse, functools, json, os, pickle, time
import jax, numpy as np
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from walk_env import HydroWalk, domain_randomize

ap = argparse.ArgumentParser()
ap.add_argument('--steps', type=float, default=3e8)
ap.add_argument('--envs', type=int, default=4096)
ap.add_argument('--out', default='runs/walk')
ap.add_argument('--smoke', action='store_true')
ap.add_argument('--resume', default=None, help='params.pkl to continue from')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
env = HydroWalk(); eval_env = HydroWalk()

cfg = dict(num_timesteps=int(a.steps), num_envs=a.envs, episode_length=1000, num_evals=20, unroll_length=20, batch_size=256, num_minibatches=32,
           num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=5e-3, discounting=0.97, reward_scaling=1.0, clipping_epsilon=0.2,
           max_grad_norm=1.0, normalize_observations=True, action_repeat=1, num_eval_envs=128, seed=1)
if a.smoke:
    cfg.update(num_timesteps=4 * 20 * 8 * 2, num_envs=8, batch_size=8, num_minibatches=2, num_evals=2, num_eval_envs=2, episode_length=40, unroll_length=20)
net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 256, 128),
                        policy_obs_key='state', value_obs_key='privileged_state')
t0 = time.time(); log = []
def progress(step, metrics):
    r = {k: float(v) for k, v in metrics.items() if np.ndim(v) == 0}
    r['step'] = int(step); r['wall_min'] = (time.time() - t0) / 60; log.append(r)
    print(f"[{r['wall_min']:6.1f} min] step {step:>11,}  reward {r.get('eval/episode_reward', float('nan')):8.2f}  "
          f"vel {r.get('eval/episode_r_vel', float('nan')):7.2f}  len {r.get('eval/avg_episode_length', float('nan')):6.1f}  "
          f"power {r.get('eval/episode_power_W', float('nan')) / max(r.get('eval/avg_episode_length', 1), 1):7.0f} W", flush=True)
    json.dump(log, open(f'{a.out}/log.json', 'w'), indent=1)
def save(step, make_policy, params):
    pickle.dump(params, open(f'{a.out}/params.pkl', 'wb'))
restore = pickle.load(open(a.resume, 'rb')) if a.resume else None
make_inference, params, metrics = ppo.train(environment=env, eval_env=eval_env, network_factory=net, randomization_fn=domain_randomize,
                                            progress_fn=progress, policy_params_fn=save, restore_params=restore, **cfg)
pickle.dump(params, open(f'{a.out}/params.pkl', 'wb'))
json.dump(dict(cfg=cfg, policy_hidden=(512, 256, 128)), open(f'{a.out}/config.json', 'w'), indent=1)
print('done in %.1f min ->' % ((time.time() - t0) / 60), a.out)
