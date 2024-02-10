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

from gymnasium import spaces
import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.sac.sac_utils import SACAgent, SACBuffer
from safe_control_gym.envs.env_wrappers.record_episode_statistics import (RecordEpisodeStatistics,
                                                                          VecRecordEpisodeStatistics)
from safe_control_gym.envs.env_wrappers.vectorized_env import make_vec_envs
from safe_control_gym.math_and_models.normalization import (BaseNormalizer, MeanStdNormalizer,
                                                            RewardStdNormalizer)
from safe_control_gym.utils.logging import ExperimentLogger
from safe_control_gym.utils.utils import get_random_state, set_random_state


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
        self.obs_space = spaces.Box(low=np.array([-2,-2,-2,-2,-0.5,-0.5,-2,-2,-2,-2]), high=np.array([2,2,2,2,0.5,0.5,2,2,2,2]), shape=(10,))
        self.act_space = spaces.Box(low=np.array([-0.25,-0.25]), high=np.array([0.25,0.25]), shape=(2,))
        self.agent = SACAgent(self.obs_space,
                              self.act_space,
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
            obs, info = self.env_reset(self.firmware_wrapper, self.use_safe_reset)
            self.info = info
            self.obs = np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :])
            self.firmware_action = self.eval_env.U_GOAL
            self.buffer = SACBuffer(self.obs_space, self.act_space, self.max_buffer_size, self.train_batch_size)
            self.train_results = {
                'total_rew': 0,
                'total_violations': 0,
                'total_mse': 0,
                'ep_lengths': [],
                'ep_returns': [],
                'ep_violations': [],
                'ep_mse': []
                }
        else:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.firmware_wrapper.close()
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
                eval_results = self.run(n_episodes=self.eval_batch_size)
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

                # Reset
                obs, info = self.env_reset(self.firmware_wrapper, self.use_safe_reset)
                self.info = info
                self.obs = np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :])
                self.firmware_action = self.eval_env.U_GOAL

                self.train_results['total_rew'] = 0
                self.train_results['total_mse'] = 0
                self.train_results['total_violations'] = 0

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
            extended_obs = np.concatenate((np.squeeze(obs), self.X_GOAL[info['current_step']//20, :]))
            obs = torch.FloatTensor(extended_obs).to(self.device)
            action = self.agent.ac.act(obs, deterministic=True)

        action = np.clip(action, [-0.25, -0.25], [0.25, 0.25])

        return action

    def run(self, n_episodes=10):
        '''Runs evaluation with current policy.'''
        obs, info = self.env_reset(self.firmware_wrapper, True)
        obs = np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :])
        firmware_action = self.eval_env.U_GOAL
        total_rew, total_violations, total_mse = 0, 0, 0
        ep_lengths, ep_returns, ep_violations, ep_mse = [], [], [], []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)

            # Adding safety filter
            certified_action, success = self.safety_filter.certify_action(obs, action, info)
            if success:
                action = certified_action
            else:
                self.safety_filter.ocp_solver.reset()
                certified_action, success = self.safety_filter.certify_action(obs, action, info)
                if success:
                    action = certified_action

            curr_time = info['current_step']//20 * self.CTRL_DT
            self.firmware_wrapper.sendCmdVel(action[0], action[1], 0, 0, curr_time)

            # Step the environment.
            next_obs, _, done, info, firmware_action = self.firmware_wrapper.step(curr_time, firmware_action)
            next_obs = np.squeeze(next_obs.reshape((12, 1))[[0,1,2,3,6,7], :])
            total_violations += np.sum(np.abs(next_obs) > [0.75, 1, 0.75, 1, 0.5, 0.5])
            rew, mse = self.get_reward(next_obs, info)

            total_rew += rew
            total_mse += mse

            if done:
                ep_lengths.append(info["current_step"]//20)
                ep_returns.append(total_rew)
                ep_violations.append(total_violations)
                ep_mse.append(total_mse/ep_lengths[-1])
                obs, info = self.env_reset(self.firmware_wrapper, True)
                obs = np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :])
                firmware_action = self.eval_env.U_GOAL
                total_rew, total_violations, total_mse = 0, 0, 0
                continue

            obs = next_obs

        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        ep_violations = np.asarray(ep_violations)
        ep_mse = np.asarray(ep_mse)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths, 'ep_violations': ep_violations, 'ep_mse': ep_mse}
        return eval_results

    def train_step(self, **kwargs):
        '''Performs a training step.'''
        self.agent.train()
        obs = self.obs
        info = self.info
        start = time.time()

        extended_obs = np.concatenate((obs, self.X_GOAL[info['current_step']//20, :])).reshape((1, -1))

        if self.total_steps < self.warm_up_steps:
            unsafe_action = np.random.rand(2)*2.0-1.0
        else:
            with torch.no_grad():
                unsafe_action = self.agent.ac.act(torch.FloatTensor(extended_obs).to(self.device), deterministic=False)

        unsafe_action = np.squeeze(unsafe_action)

        # Adding safety filter
        applied_action = unsafe_action
        success = False

        if self.safety_filter is not None and (self.filter_train_actions is True or self.penalize_sf_diff is True):
            certified_action, success = self.safety_filter.certify_action(obs, applied_action, info)
            if success and self.filter_train_actions is True:
                applied_action = certified_action
            else:
                self.safety_filter.ocp_solver.reset()
                certified_action, success = self.safety_filter.certify_action(obs, applied_action, info)
                if success and self.filter_train_actions is True:
                    applied_action = certified_action

        curr_time = info['current_step']//20 * self.CTRL_DT
        self.firmware_wrapper.sendCmdVel(applied_action[0], applied_action[1], 0, 0, curr_time)

        # Step the environment.
        next_obs, _, done, info, self.firmware_action = self.firmware_wrapper.step(curr_time, self.firmware_action)
        next_obs = np.squeeze(next_obs.reshape((12, 1))[[0,1,2,3,6,7], :])
        self.train_results['total_violations'] += np.sum(np.abs(next_obs) > [0.75, 1, 0.75, 1, 0.5, 0.5])
        next_i = min(info['current_step']//20, self.X_GOAL.shape[0]-1)
        extended_next_obs = np.concatenate((next_obs, self.X_GOAL[next_i, :])).reshape((1, -1))
        rew, mse = self.get_reward(next_obs, info)

        if self.penalize_sf_diff and success:
            unsafe_rew = np.log(rew)
            unsafe_rew -= self.sf_penalty * np.linalg.norm(np.clip(unsafe_action, [-0.25, -0.25], [0.25, 0.25]) - certified_action)
            unsafe_rew = np.exp(unsafe_rew)
        else:
            unsafe_rew = rew

        # Constraint Penalty
        if self.firmware_wrapper.env.use_constraint_penalty and np.any(np.abs(next_obs) > [0.75, 1, 0.75, 1, 0.5, 0.5]):
            unsafe_rew = np.log(rew)
            unsafe_rew -= 1.0
            unsafe_rew = np.exp(unsafe_rew)

        self.train_results['total_rew'] += rew
        self.train_results['total_mse'] += mse

        mask = int(1 - done)

        if done:
            if info['current_step']//20 >= self.X_GOAL.shape[0]:
                mask = 1
            self.train_results['ep_lengths'].append(info["current_step"]//20)
            self.train_results['ep_returns'].append(self.train_results['total_rew'])
            self.train_results['ep_violations'].append(self.train_results['total_violations'])
            self.train_results['ep_mse'].append(self.train_results['total_mse']/self.train_results['ep_lengths'][-1])

            self.train_results['total_rew'] = 0
            self.train_results['total_mse'] = 0
            self.train_results['total_violations'] = 0

            next_obs, info = self.env_reset(self.firmware_wrapper, self.use_safe_reset)
            next_obs = np.squeeze(next_obs.reshape((12, 1))[[0,1,2,3,6,7], :])
            self.firmware_action = self.eval_env.U_GOAL

        self.buffer.push({
            'obs': extended_obs,
            'act': unsafe_action,
            'rew': unsafe_rew,
            # 'next_obs': next_obs,
            # 'mask': mask,
            'next_obs': extended_next_obs,
            'mask': mask,
        })

        self.obs = next_obs
        self.info = info
        self.total_steps += self.rollout_batch_size

        # learn
        results = defaultdict(list)
        if self.total_steps > self.warm_up_steps and not self.total_steps % self.train_interval:
            for _ in range(self.train_interval):
                batch = self.buffer.sample(self.train_batch_size, self.device)
                res = self.agent.update(batch)
                for k, v in res.items():
                    results[k].append(v)

        results = {k: sum(v) / len(v) for k, v in results.items()}
        results.update(self.train_results)
        results.update({'step': self.total_steps, 'elapsed_time': time.time() - start})
        return results

    def get_reward(self, obs, info):
        wp_idx = min(info['current_step']//20, self.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
        state_error = obs[:4] - self.X_GOAL[wp_idx]
        dist = np.sum(np.array([2, 0, 2, 0]) * state_error * state_error)
        rew = -dist
        rew = np.exp(rew)

        mse = np.sum(np.array([1, 1, 1, 1]) * state_error * state_error)

        return rew, mse

    def log_step(self,
                 results
                 ):
        '''Does logging after a training step.'''
        step = results['step']
        # runner stats
        self.logger.add_scalars(
            {
                'step': step,
                'step_time': results['elapsed_time'],
                'progress': step / self.max_env_steps
            },
            step,
            prefix='time',
            write=False,
            write_tb=False)
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['policy_loss', 'critic_loss', 'entropy_loss']
            },
            step,
            prefix='loss')
        # Performance stats.
        ep_lengths = np.array(results['ep_lengths'])
        ep_returns = np.array(results['ep_returns'])
        ep_constraint_violation = np.array(results['ep_violations'])
        self.logger.add_scalars(
            {
                'ep_length': ep_lengths.mean(),
                'ep_return': ep_returns.mean(),
                'ep_reward': (ep_returns / ep_lengths).mean(),
                'ep_constraint_violation': ep_constraint_violation.mean()
            },
            step,
            prefix='stat')
        # Total constraint violation during learning.
        if 'eval' in results:
            eval_ep_lengths = results['eval']['ep_lengths']
            eval_ep_returns = results['eval']['ep_returns']
            eval_constraint_violation = results['eval']['ep_violations']
            eval_mse = results['eval']['ep_mse']
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
        # Print summary table
        self.logger.dump_scalars()
        self.train_results = {
                'total_rew': 0,
                'total_violations': 0,
                'total_mse': 0,
                'ep_lengths': [],
                'ep_returns': [],
                'ep_violations': [],
                'ep_mse': []
                }

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
        act = np.array([0,0])
        obs, info = env.reset()

        if self.safety_filter is not None:
            self.safety_filter.reset_before_run()

        if use_safe_reset is True and self.safety_filter is not None:
            while success is not True or np.any(self.safety_filter.slack_prev > 1e-4):
                obs, info = env.reset()
                info['current_step'] = 1
                self.safety_filter.reset_before_run()
                _, success = self.safety_filter.certify_action(np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :]), act, info)
                if not success:
                    self.safety_filter.ocp_solver.reset()
                    _, success = self.safety_filter.certify_action(np.squeeze(obs.reshape((12, 1))[[0,1,2,3,6,7], :]), act, info)

        return obs, info
