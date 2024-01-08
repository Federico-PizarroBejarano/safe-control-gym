'''Soft Actor Critic (SAC)

Adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py

References papers & code:
    * [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
    * [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
    * [Soft Actor-Critic for Discrete Action Settings](https://arxiv.org/pdf/1910.07207.pdf)
    * [openai spinning up - sac](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac)
    * [rlkit - sac](https://github.com/vitchyr/rlkit/tree/7daf34b0ef2277d545a0ee792399a2ae6c3fb6ad/rlkit/torch/sac)
    * [ray rllib - sac](https://github.com/ray-project/ray/tree/master/rllib/agents/sac)
    * [curl - curl_sac](https://github.com/MishaLaskin/curl/blob/master/curl_sac.py)
'''

import os
import time
from collections import defaultdict

import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env_utils import _flatten_obs, _unflatten_obs
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state


class SAC(BaseController):
    '''soft actor critic.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        self.filter_train_actions = False
        self.penalize_sf_diff = False
        self.sf_penalty = 1
        self.use_safe_reset = False
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)

        # task
        if self.training:
            # training (+ evaluation)
            self.env = make_vec_envs(env_func, None, self.rollout_batch_size, self.num_workers, seed)
            self.env = VecRecordEpisodeStatistics(self.env, self.deque_size)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, self.deque_size)
            self.model = self.get_prior(self.eval_env, self.prior_info)
        else:
            # testing only
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)

        # agent
        self.agent = SACAgent(self.env.observation_space,
                              self.env.action_space,
                              hidden_dim=self.hidden_dim,
                              gamma=self.gamma,
                              tau=self.tau,
                              init_temperature=self.init_temperature,
                              use_entropy_tuning=self.use_entropy_tuning,
                              target_entropy=self.target_entropy,
                              actor_lr=self.actor_lr,
                              critic_lr=self.critic_lr,
                              entropy_lr=self.entropy_lr,
                              activation=self.activation)
        self.agent.to(self.device)

        # pre-/post-processing
        self.obs_normalizer = BaseNormalizer()
        if self.norm_obs:
            self.obs_normalizer = MeanStdNormalizer(shape=self.env.observation_space.shape, clip=self.clip_obs, epsilon=1e-8)

        self.reward_normalizer = BaseNormalizer()
        if self.norm_reward:
            self.reward_normalizer = RewardStdNormalizer(gamma=self.gamma, clip=self.clip_reward, epsilon=1e-8)

        # logging
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # disable logging to texts and tfboard for testing
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

        # Adding safety filter
        self.safety_filter = None

    def reset(self):
        '''Prepares for training or testing.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.total_steps = 0
            obs, info = self.env_reset(self.env)
            self.info = info['n'][0]
            self.true_obs = obs
            self.obs = self.obs_normalizer(obs)
            self.buffer = SACBuffer(self.env.observation_space, self.env.action_space, self.max_buffer_size, self.train_batch_size)
        else:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def save(self, path, save_buffer=False):
        '''Saves model params and experiment state to checkpoint path.'''
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir, exist_ok=True)

        state_dict = {
            'agent': self.agent.state_dict(),
            'obs_normalizer': self.obs_normalizer.state_dict(),
            'reward_normalizer': self.reward_normalizer.state_dict()
        }
        if self.training:
            exp_state = {
                'total_steps': self.total_steps,
                'obs': self.obs,
                'random_state': get_random_state(),
                'env_random_state': self.env.get_env_random_state()
            }
            # latest checkpoint shoud enable save_buffer (for experiment restore),
            # but intermediate checkpoint shoud not, to save storage (buffer is large)
            if save_buffer:
                exp_state['buffer'] = self.buffer.state_dict()
            state_dict.update(exp_state)
        torch.save(state_dict, path)

    def load(self, path):
        '''Restores model and experiment given checkpoint path.'''
        state = torch.load(path)

        # restore params
        self.agent.load_state_dict(state['agent'])
        self.obs_normalizer.load_state_dict(state['obs_normalizer'])
        self.reward_normalizer.load_state_dict(state['reward_normalizer'])

        # restore experiment state
        if self.training:
            self.total_steps = state['total_steps']
            self.obs = state['obs']
            set_random_state(state['random_state'])
            self.env.set_env_random_state(state['env_random_state'])
            if 'buffer' in state:
                self.buffer.load_state_dict(state['buffer'])
            self.logger.load(self.total_steps)

    def learn(self, env=None, **kwargs):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        while self.total_steps < self.max_env_steps:
            results = self.train_step()

            # checkpoint
            if self.total_steps >= self.max_env_steps or (self.save_interval and self.total_steps % self.save_interval == 0):
                # latest/final checkpoint
                self.save(self.checkpoint_path, save_buffer=False)
                self.logger.info(f'Checkpoint | {self.checkpoint_path}')
            if self.num_checkpoints and self.total_steps % (self.max_env_steps // self.num_checkpoints) == 0:
                # intermediate checkpoint
                path = os.path.join(self.output_dir, 'checkpoints', f'model_{self.total_steps}.pt')
                self.save(path, save_buffer=True)

            # eval
            if self.eval_interval and self.total_steps % self.eval_interval == 0:
                eval_results = self.run(env=self.eval_env, n_episodes=self.eval_batch_size)
                results['eval'] = eval_results
                self.logger.info('Eval | ep_lengths {:.2f} +/- {:.2f} | ep_return {:.3f} +/- {:.3f}'.format(eval_results['ep_lengths'].mean(),
                                                                                                            eval_results['ep_lengths'].std(),
                                                                                                            eval_results['ep_returns'].mean(),
                                                                                                            eval_results['ep_returns'].std()))
                # save best model
                eval_score = eval_results['ep_returns'].mean()
                eval_best_score = getattr(self, 'eval_best_score', -np.infty)
                if self.eval_save_best and eval_best_score < eval_score:
                    self.eval_best_score = eval_score
                    self.save(os.path.join(self.output_dir, 'model_best.pt'), save_buffer=False)

            # logging
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
            action = self.agent.ac.act(obs, deterministic=True)

        return action

    def run(self, env=None, render=False, n_episodes=10, verbose=False, **kwargs):
        '''Runs evaluation with current policy.'''
        self.agent.eval()
        self.obs_normalizer.set_read_only()
        if env is None:
            env = self.env
        else:
            if not is_wrapped(env, RecordEpisodeStatistics):
                env = RecordEpisodeStatistics(env, n_episodes)
                # Add episodic stats to be tracked.
                env.add_tracker('constraint_violation', 0, mode='queue')
                env.add_tracker('constraint_values', 0, mode='queue')
                env.add_tracker('mse', 0, mode='queue')

        obs, info = self.env_reset(env)
        true_obs = obs
        obs = self.obs_normalizer(obs)
        ep_returns, ep_lengths = [], []
        frames = []
        total_return = 0

        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)

            # Adding safety filter
            applied_action = action
            success = False

            physical_action = env.denormalize_action(action)
            unextended_obs = np.squeeze(true_obs)[:env.symbolic.nx]
            certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
            if success:
                applied_action = env.normalize_action(certified_action)
            else:
                self.safety_filter.ocp_solver.reset()
                certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
                if success:
                    applied_action = self.env.envs[0].normalize_action(certified_action)

            action = np.atleast_2d(np.squeeze([applied_action]))
            obs, rew, done, info = env.step(action)
            total_return += rew

            if render:
                env.render()
                frames.append(env.render('rgb_array'))
            if verbose:
                print(f'obs {obs} | act {action}')

            if done:
                assert 'episode' in info
                ep_returns.append(total_return)
                ep_lengths.append(info['episode']['l'])
                obs, info = self.env_reset(env)
                total_return = 0
            true_obs = obs
            obs = self.obs_normalizer(obs)

        # collect evaluation results
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
        if len(frames) > 0:
            eval_results['frames'] = frames
        # Other episodic stats from evaluation env.
        if len(env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

    def train_step(self, **kwargs):
        '''Performs a training step.'''
        self.agent.train()
        self.obs_normalizer.unset_read_only()
        obs = self.obs
        true_obs = self.true_obs
        info = self.info
        start = time.time()

        if self.total_steps < self.warm_up_steps:
            action = np.stack([self.env.action_space.sample() for _ in range(self.rollout_batch_size)])
        else:
            with torch.no_grad():
                action = self.agent.ac.act(torch.FloatTensor(obs).to(self.device), deterministic=False)

        # Adding safety filter
        unsafe_action = action
        applied_action = action
        success = False

        if self.safety_filter is not None and (self.filter_train_actions is True or self.penalize_sf_diff is True):
            physical_action = self.env.envs[0].denormalize_action(action)
            unextended_obs = np.squeeze(true_obs)[:self.env.envs[0].symbolic.nx]
            certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
            if success and self.filter_train_actions is True:
                applied_action = self.env.envs[0].normalize_action(certified_action)
            else:
                self.safety_filter.ocp_solver.reset()
                certified_action, success = self.safety_filter.certify_action(unextended_obs, physical_action, info)
                if success and self.filter_train_actions is True:
                    applied_action = self.env.envs[0].normalize_action(certified_action)

        action = np.atleast_2d(np.squeeze([applied_action]))
        next_obs, rew, done, info = self.env.step(action)
        if done[0] and self.use_safe_reset is True:
            next_obs, info = self.env_reset(self.env)
        if self.penalize_sf_diff and success:
            unsafe_rew = np.log(rew)
            unsafe_rew -= self.sf_penalty * np.linalg.norm(physical_action - certified_action)
            unsafe_rew = np.exp(unsafe_rew)
        else:
            unsafe_rew = rew
        next_true_obs = next_obs
        next_obs = self.obs_normalizer(next_obs)
        rew = self.reward_normalizer(rew, done)
        mask = 1 - np.asarray(done)

        # time truncation is not true termination
        terminal_idx, terminal_obs = [], []
        for idx, inf in enumerate(info['n']):
            if 'terminal_info' not in inf:
                continue
            inff = inf['terminal_info']
            if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                terminal_idx.append(idx)
                terminal_obs.append(inf['terminal_observation'])
        if len(terminal_obs) > 0:
            terminal_obs = _unflatten_obs(self.obs_normalizer(_flatten_obs(terminal_obs)))

        # collect the true next states and masks (accounting for time truncation)
        true_next_obs = _unflatten_obs(next_obs)
        true_mask = mask.copy()
        for idx, term_ob in zip(terminal_idx, terminal_obs):
            true_next_obs[idx] = term_ob
            true_mask[idx] = 1.0
        true_next_obs = _flatten_obs(true_next_obs)

        if not np.array_equal(unsafe_rew, rew):
            self.buffer.push({
                'obs': obs,
                'act': unsafe_action,
                'rew': unsafe_rew,
                # 'next_obs': next_obs,
                # 'mask': mask,
                'next_obs': true_next_obs,
                'mask': true_mask,
            })

        self.buffer.push({
            'obs': obs,
            'act': applied_action,
            'rew': rew,
            # 'next_obs': next_obs,
            # 'mask': mask,
            'next_obs': true_next_obs,
            'mask': true_mask,
        })
        obs = next_obs
        true_obs = next_true_obs
        info = info['n'][0]

        self.obs = obs
        self.true_obs = true_obs
        self.info = info
        self.total_steps += self.rollout_batch_size

        # learn
        results = defaultdict(list)
        if self.total_steps > self.warm_up_steps and not self.total_steps % self.train_interval:
            # Regardless of how long you wait between updates,
            # the ratio of env steps to gradient steps is locked to 1.
            # alternatively, can update once each step
            for _ in range(self.train_interval):
                batch = self.buffer.sample(self.train_batch_size, self.device)
                res = self.agent.update(batch)
                for k, v in res.items():
                    results[k].append(v)

        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def log_step(self, results):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'time': results['elapsed_time'],
                'progress': step / self.max_env_steps,
            },
            step,
            prefix='time',
            write=False,
            write_tb=False)

        # learning stats
        if 'policy_loss' in results:
            self.logger.add_scalars(
                {
                    k: results[k]
                    for k in ['policy_loss', 'critic_loss', 'entropy_loss']
                },
                step,
                prefix='loss')

        # performance stats
        ep_lengths = np.asarray(self.env.length_queue)
        ep_returns = np.asarray(self.env.return_queue)
        ep_constraint_violation = np.asarray(self.env.queued_stats['constraint_violation'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')

        # total constraint violation during learning
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
                    'mse': eval_mse.mean()
                },
                step,
                prefix='stat_eval')

        # print summary table
        self.logger.dump_scalars()

    def env_reset(self, env):
        '''Resets the environment until a feasible initial state is found.

        Args:
            env (BenchmarkEnv): The environment that is being reset.

        Returns:
            obs (ndarray): The initial observation.
            info (dict): The initial info.
        '''
        success = False
        action = self.model.U_EQ
        obs, info = env.reset()
        if self.safety_filter is not None:
            self.safety_filter.reset_before_run()

        if self.use_safe_reset is True and self.safety_filter is not None:
            while success is not True or np.any(self.safety_filter.slack_prev > 1e-4):
                obs, info = env.reset()
                info['current_step'] = 1
                unextended_obs = np.squeeze(obs)[:self.env.envs[0].symbolic.nx]
                self.safety_filter.reset_before_run()
                _, success = self.safety_filter.certify_action(unextended_obs, action, info)
                if not success:
                    self.safety_filter.ocp_solver.reset()
                    _, success = self.safety_filter.certify_action(unextended_obs, action, info)

        return obs, info
