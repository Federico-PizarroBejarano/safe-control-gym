'''Proximal Policy Optimization (PPO)

Based on:
    * https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
    * (hyperparameters) https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml

Additional references:
    * Proximal Policy Optimization Algorithms - https://arxiv.org/pdf/1707.06347.pdf
    * Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO - https://arxiv.org/pdf/2005.12729.pdf
    * pytorch-a2c-ppo-acktr-gail - https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
    * openai spinning up - ppo - https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo
    * stable baselines3 - ppo - https://github.com/DLR-RM/stable-baselines3/tree/master/stable_baselines3/ppo
'''

import os
import time

import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.ppo.ppo_utils import PPOAgent, PPOBuffer, compute_returns_and_advantages
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import is_wrapped


class PPO(BaseController):
    '''Proximal policy optimization.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=True,
                 seed=0,
                 **kwargs):
        self.filter_train_actions = True
        self.penalize_sf_diff = False
        self.sf_penalty = 1
        self.use_safe_reset = False
        self.use_adv_reward = True
        self.adv_reward_temperature = kwargs.get('adv_reward_temperature', 50.0)
        self.adv_reward_scale = kwargs.get('adv_reward_scale', 100.0)
        self.adv_reward_exponential = kwargs.get('adv_reward_exponential', True)
        self.adv_correction_weight = kwargs.get('adv_correction_weight', 1.0)
        self.sf_idle_floor = kwargs.get('sf_idle_floor', 0.0)
        self._sf_idle_penalty_weight = kwargs.get('sf_idle_penalty_weight', 0.0)
        self._sf_idle_penalty = 0.0
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        # Task.
        if self.training:
            # Training and testing.
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
            self.model = self.get_prior(self.eval_env, self.prior_info)
        else:
            # Testing only.
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)
        # Agent.
        self.agent = PPOAgent(self.env.observation_space,
                              self.env.action_space,
                              hidden_dim=self.hidden_dim,
                              use_clipped_value=self.use_clipped_value,
                              clip_param=self.clip_param,
                              target_kl=self.target_kl,
                              entropy_coef=self.entropy_coef,
                              actor_lr=self.actor_lr,
                              critic_lr=self.critic_lr,
                              opt_epochs=self.opt_epochs,
                              mini_batch_size=self.mini_batch_size,
                              activation=self.activation)
        self.agent.to(self.device)
        # Pre-/post-processing.
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.env.observation_space.shape, clip=self.clip_obs, epsilon=1e-8)
        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)
        # Logging.
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # Disable logging to file and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

        # Adding safety filter
        self.safety_filter = None

    def reset(self):
        '''Do initializations for training or evaluation.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, info = self.env_reset(self.env, self.use_safe_reset)
            self.info = info['n'][0]
            self.obs = self.obs_normalizer(obs)
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self,
             path,
             ):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)
        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict(),
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'env_random_state': self.env.get_env_random_state()
            }
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self,
             path,
             ):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path)
        # Restore policy.
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])
        # Restore experiment state.
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            self.env.set_env_random_state(state['env_random_state'])
            self.logger.load(self.total_steps)

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        while self.total_steps < self.max_env_steps:
            results = self.train_step()
            # Checkpoint.
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # Latest/final checkpoint.
                self.save(self.checkpoint_path)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
            if self.num_checkpoints and self.total_steps % (self.max_env_steps // self.num_checkpoints) == 0:
                # Intermediate checkpoint.
                path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                self.save(path)
            # Evaluation.
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                                                            eval_results['ep_lengths'].std(),
                                                                                                            eval_results['ep_returns'].mean(),
                                                                                                            eval_results['ep_returns'].std()))
                # Save best model.
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'))
            # Logging.
            if self.log_interval and self.total_steps % self.log_interval == 0:
                self.log_step(results)

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            action = self.agent.ac.act(obs)

        return action

    def run(self,
            env=None,
            render=False,
            n_episodes=10,
            verbose=False,
            ):
        '''Runs evaluation with current policy.'''
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        self.use_adv_reward = True
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = self.env_reset(env, True)
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        total_return = 0
        start = time.time()
        while len(ep_returns) < n_episodes:
            prev_obs = obs
            action = self.select_action(obs=obs, info=info)

            # Adding safety filter
            physical_action = None
            certified_action = None
            success = False
            if self.safety_filter is not None:
                physical_action = env.denormalize_action(action)
                unextended_obs = np.squeeze(obs)[:env.symbolic.nx]
                certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
                if success:
                    action = env.normalize_action(certified_action)
                elif self.safety_filter.use_acados:
                    self.safety_filter.ocp_solver.reset()

            action = np.atleast_2d(np.squeeze([action]))
            next_obs, rew, done, info = env.step(action)
            if self.use_adv_reward:
                uncert_act = physical_action if self.safety_filter is not None else None
                cert_act = certified_action if (self.safety_filter is not None and success) else None
                rew = self.adversarial_reward(prev_obs, next_obs, action, info,
                                              uncert_action=uncert_act, cert_action=cert_act)
            total_return += rew

            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {next_obs} | act {action}')
            if done:
                assert 'episode' in info
                ep_returns.append(total_return)
                ep_lengths.append(info['episode']['l'])
                obs, info = self.env_reset(env, True)
                total_return = 0
            else:
                obs = next_obs
            obs = self.obs_normalizer(obs)
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {
            'ep_returns': ep_returns,
            'ep_lengths': ep_lengths,
            'elapsed_time': time.time() - start
        }
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def train_step(self):
        '''Performs a training/fine-tuning step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        rollouts = PPOBuffer(self.env.observation_space, self.env.action_space, self.rollout_steps, self.rollout_batch_size)
        obs = self.obs
        info = self.info
        start = time.time()
        for _ in range(self.rollout_steps):
            with torch.no_grad():
                action, v, logp = self.agent.ac.step(torch.FloatTensor(obs).to(self.device))
                unsafe_action = action

            action_np = np.array(action, copy=False).reshape(self.rollout_batch_size, -1)

            # Adding safety filter
            success = False
            if self.safety_filter is not None and (self.filter_train_actions is True or self.penalize_sf_diff is True):
                uncertified = action_np[0, :].copy()
                # physical_action = self.env.envs[0].denormalize_action(action)
                physical_action = self.env.envs[0].denormalize_action(uncertified)
                unextended_obs = np.squeeze(obs)[:self.env.envs[0].symbolic.nx]
                certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
                if success and self.filter_train_actions is True:
                    # action = self.env.envs[0].normalize_action(certified_action)
                    action_np[0, :] = self.env.envs[0].normalize_action(np.atleast_1d(certified_action))
                elif not success and self.safety_filter.use_acados:
                    self.safety_filter.ocp_solver.reset()

                if success and np.linalg.norm(physical_action) < self.sf_idle_floor:
                    self._sf_idle_penalty = self._sf_idle_penalty_weight
                else:
                    self._sf_idle_penalty = 0.0
            else:
                self._sf_idle_penalty = 0.0

            action = action_np
            # action = np.atleast_2d(np.squeeze([action])).reshape((self.rollout_batch_size, -1))
            next_obs, rew, done, info = self.env.step(action)
            if done[0] and self.use_safe_reset:
                prev_info = info['n'][0]
                next_obs, info = self.env_reset(self.env, self.use_safe_reset)
                info['n'][0]['terminal_info'] = prev_info['terminal_info']
                info['n'][0]['terminal_observation'] = prev_info['terminal_observation']
            # if self.penalize_sf_diff and success:
            #     rew = np.log(rew)
            #     rew -= self.sf_penalty * np.linalg.norm(physical_action - certified_action)
            #     rew = np.exp(rew)
            next_obs = self.obs_normalizer(next_obs)
            rew = self.reward_normalizer(rew, done)
            if self.use_adv_reward:
                # Pass uncertified and certified actions to adversarial reward
                uncert_act = physical_action if self.safety_filter is not None else None
                cert_act = certified_action if (self.safety_filter is not None and success) else None
                rew = self.adversarial_reward(obs, next_obs, action, info,
                                              uncert_action=uncert_act,
                                              cert_action=cert_act)
                adv_stats = {}
                raw_stats = getattr(self, '_adv_reward_raw_stats', None)
                scaled_stats = getattr(self, '_adv_reward_scaled_stats', None)
                if raw_stats:
                    adv_stats.update({f'raw_{k}': raw_stats[k] for k in ['mean', 'min', 'max']})
                if scaled_stats:
                    adv_stats.update({f'scaled_{k}': scaled_stats[k] for k in ['mean', 'min', 'max']})
                if adv_stats:
                    adv_stats['temperature'] = float(self.adv_reward_temperature)
                self._adv_reward_last_stats = adv_stats
            else:
                self._adv_reward_last_stats = {}
            mask = 1 - done.astype(float)
            # Time truncation is not the same as true termination.
            terminal_v = np.zeros_like(v)
            for idx, inf in enumerate(info['n']):
                if 'terminal_info' not in inf:
                    continue
                inff = inf['terminal_info']
                if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                    terminal_obs = inf['terminal_observation']
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                    terminal_v[idx] = terminal_val

            rollouts.push({'obs': obs, 'act': unsafe_action, 'rew': rew, 'mask': mask, 'v': v, 'logp': logp, 'terminal_v': terminal_v})
            obs = next_obs
            info = info['n'][0]
        self.obs = obs
        self.info = info
        self.total_steps += self.rollout_batch_size * self.rollout_steps
        # Learn from rollout batch.
        last_val = self.agent.ac.critic(torch.FloatTensor(obs).to(self.device)).detach().cpu().numpy()
        ret, adv = compute_returns_and_advantages(rollouts.rew,
                                                  rollouts.v,
                                                  rollouts.mask,
                                                  rollouts.terminal_v,
                                                  last_val,
                                                  gamma=self.gamma,
                                                  use_gae=self.use_gae,
                                                  gae_lambda=self.gae_lambda)
        rollouts.ret = ret
        # Prevent divide-by-0 for repetitive tasks.
        rollouts.adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        results = self.agent.update(rollouts, self.device)
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        if getattr(self, '_adv_reward_last_stats', None):
            results.update({f'adv_reward_{k}': v for k, v in self._adv_reward_last_stats.items()})
        return results

    def log_step(self,
                 results
                 ):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'progress': step / self.max_env_steps,
            },
            step,
            prefix='time',
            write=False,
            write_tb=False)
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl']
            },
            step,
            prefix='loss')
        # Performance stats.
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean(),
                'step_time': results['elapsed_time'],
            },
            step,
            prefix='stat')
        # Total constraint violation during learning.
        total_violations = self.env.accumulated_stats['constraint_violation']
        self.logger.add_scalars({'constraint_violation': total_violations}, step, prefix='stat')
        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['constraint_violation']
            eval_mse = results['eval']['mse']
            self.logger.add_scalars(
                {
                    'ep_length': eval_ep_lengths.mean(),
                    'ep_return': eval_ep_returns.mean(),
                    'ep_reward': (eval_ep_returns / eval_ep_lengths).mean(),
                    'constraint_violation': eval_constraint_violation.mean(),
                    'mse': eval_mse.mean(),
                    'step_time': results['eval']['elapsed_time'],
                },
                step,
                prefix='stat_eval')
        adv_keys = [k for k in results.keys() if k.startswith('adv_reward_')]
        if adv_keys:
            self.logger.add_scalars({k: results[k] for k in adv_keys}, step, prefix='adv_reward')
        # Print summary table
        self.logger.dump_scalars()

    def env_reset(self, env, use_safe_reset):
        '''Resets the environment until a feasible initial state is found.

        Args:
            env (BenchmarkEnv): The environment that is being reset.
            use_safe_reset (bool): Whether to safely reset the system using the SF.

        Returns:
            obs (ndarray): The initial observation.
            info (dict): The initial info.
        '''
        success = False
        action = self.model.U_EQ
        obs, info = env.reset()
        if self.safety_filter is not None:
            self.safety_filter.reset_before_run()

        if use_safe_reset is True and self.safety_filter is not None:
            while success is not True or np.any(self.safety_filter.slack_prev > 1e-4):
                obs, info = env.reset()
                info['current_step'] = 1
                unextended_obs = np.squeeze(obs)[:self.env.envs[0].symbolic.nx]
                self.safety_filter.reset_before_run()
                _, success = self.safety_filter.certify_action(unextended_obs, action, info)
                if not success and self.safety_filter.use_acados:
                    self.safety_filter.ocp_solver.reset()

        return obs, info

    def adversarial_reward(self, obs, next_obs, action, info, uncert_action=None, cert_action=None):
        '''Adversarial reward function for training agents that cause safety filter chattering.

        The goal is to maximize how much the safety filter has to correct agent actions
        while staying within constraints (not escaping).

        Configurable reward components (set via algo_config):
            - adv_reward_temperature: Scaling for normalization (default: 15.0)
            - adv_use_correction_reward: Enable correction-based rewards (default: True)
            - adv_use_correction_ratio: Enable correction ratio rewards (default: True)
            - adv_use_correction_bonus: Enable tiered correction bonuses (default: True)
            - adv_use_no_correction_penalty: Penalize no corrections (default: True)
            - adv_use_theta_reward: Reward high theta angles (default: True)
            - adv_use_velocity_reward: Reward angular velocity (default: True)
            - adv_use_oscillation_reward: Reward direction changes (default: True)
            - adv_use_cart_penalty: Penalize cart position (default: True)
            - adv_use_stability_penalty: Penalize being too stable (default: True)

        Returns shape (batch,).
        '''
        next_obs_b = np.atleast_2d(next_obs)
        obs_b = np.atleast_2d(obs)
        B = next_obs_b.shape[0]

        adv_rew = np.zeros(B, dtype=np.float32)
        raw_rew = np.zeros(B, dtype=np.float32)
        infos = info.get('n', [info] * B) if isinstance(info, dict) else [dict()] * B

        # Config
        temperature = getattr(self, 'adv_reward_temperature', 15.0)
        use_correction_reward = getattr(self, 'adv_use_correction_reward', True)
        use_correction_ratio = getattr(self, 'adv_use_correction_ratio', True)
        use_correction_bonus = getattr(self, 'adv_use_correction_bonus', True)
        use_no_correction_penalty = getattr(self, 'adv_use_no_correction_penalty', True)
        use_theta_reward = getattr(self, 'adv_use_theta_reward', True)
        use_velocity_reward = getattr(self, 'adv_use_velocity_reward', True)
        use_oscillation_reward = getattr(self, 'adv_use_oscillation_reward', True)
        use_cart_penalty = getattr(self, 'adv_use_cart_penalty', True)
        use_stability_penalty = getattr(self, 'adv_use_stability_penalty', True)

        # Weights
        w_correction = getattr(self, 'adv_w_correction', 25.0)
        w_correction_ratio = getattr(self, 'adv_w_correction_ratio', 40.0)
        w_correction_bonus = getattr(self, 'adv_w_correction_bonus', 20.0)
        w_no_correction_penalty = getattr(self, 'adv_w_no_correction_penalty', 15.0)
        w_theta = getattr(self, 'adv_w_theta', 15.0)
        w_velocity = getattr(self, 'adv_w_velocity', 8.0)
        w_oscillation = getattr(self, 'adv_w_oscillation', 10.0)
        w_cart_penalty = getattr(self, 'adv_w_cart_penalty', 5.0)
        w_stability_penalty = getattr(self, 'adv_w_stability_penalty', 10.0)
        w_termination_penalty = getattr(self, 'adv_w_termination_penalty', 100.0)

        for i in range(B):
            s_prev = obs_b[i]
            s = next_obs_b[i]

            if s.shape[0] >= 4:  # cartpole: [x, x_dot, theta, theta_dot, ...]
                x, _, th, thdot = s[0], s[1], s[2], s[3]
                _, thdot_prev = s_prev[2], s_prev[3]

                # Get theta constraint limit
                theta_lim = 0.2
                try:
                    theta_lim = float(self.env.envs[0].constraints.constraints[0].upper_bounds[2])
                except Exception:
                    pass
                theta_ratio = abs(th) / max(theta_lim, 1e-6)

                # Check termination
                terminated = False
                try:
                    ti = infos[i].get('terminal_info', {})
                    if ti and not ti.get('TimeLimit.truncated', False):
                        terminated = True
                except Exception:
                    pass

                if terminated:
                    r = -w_termination_penalty
                else:
                    r = 0.0

                    # Correction rewards
                    if uncert_action is not None and cert_action is not None:
                        uncert_mag = float(np.linalg.norm(np.atleast_1d(uncert_action)))
                        correction = float(np.linalg.norm(np.atleast_1d(uncert_action) - np.atleast_1d(cert_action)))

                        if use_correction_reward:
                            r += w_correction * correction

                        if use_correction_ratio and uncert_mag > 0.1:
                            r += w_correction_ratio * (correction / uncert_mag)

                        if use_correction_bonus:
                            if correction > 1.0:
                                r += w_correction_bonus
                            if correction > 2.0:
                                r += w_correction_bonus

                        if use_no_correction_penalty and correction < 0.1:
                            r -= w_no_correction_penalty

                    elif uncert_action is not None:
                        r += 5.0 * float(np.linalg.norm(np.atleast_1d(uncert_action)))

                    # State rewards
                    if use_theta_reward:
                        r += w_theta * theta_ratio

                    if use_velocity_reward:
                        r += w_velocity * min(abs(thdot), 2.0)

                    if use_oscillation_reward:
                        if np.sign(thdot) != np.sign(thdot_prev) and thdot_prev != 0:
                            r += w_oscillation

                    # Penalties
                    if use_cart_penalty and abs(x) > 1.5:
                        r -= w_cart_penalty * (abs(x) - 1.5)

                    if use_stability_penalty:
                        if theta_ratio < 0.2 and abs(thdot) < 0.5:
                            r -= w_stability_penalty

            else:
                r = 0.25 * np.linalg.norm(s) + 0.5 * np.linalg.norm(s - s_prev)

            raw_rew[i] = float(r)
            scaled = temperature * np.log(1.0 + np.exp(r / max(temperature, 1e-6)))
            adv_rew[i] = float(scaled)

        self._adv_reward_raw_stats = {'mean': float(np.mean(raw_rew)), 'min': float(np.min(raw_rew)), 'max': float(np.max(raw_rew))}
        self._adv_reward_scaled_stats = {'mean': float(np.mean(adv_rew)), 'min': float(np.min(adv_rew)), 'max': float(np.max(adv_rew))}

        return adv_rew
