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

from gymnasium import spaces
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
from safe_control_gym.utils.utils import get_random_state, is_wrapped, set_random_state


class PPO(BaseController):
    '''Proximal policy optimization.'''

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
        self.obs_space = spaces.Box(low=np.array([-2,-2,-2,-2,-2,-2,-2,-2]), high=np.array([2,2,2,2,2,2,2,2]), shape=(8,))
        self.act_space = spaces.Box(low=np.array([-0.25,-0.25]), high=np.array([0.25,0.25]), shape=(2,))
        self.agent = PPOAgent(self.obs_space,
                              self.act_space,
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
            obs, info = self.env_reset(self.env)
            self.info = info['n'][0]
            self.true_obs = obs
            self.obs = self.obs_normalizer(obs)
        else:
            # Add episodic stats to be tracked.
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
                'random_state': get_random_state(),
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
            set_random_state(state['random_state'])
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
                eval_results = self.run(n_episodes=self.eval_batch_size)
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
            extended_obs = np.concatenate((np.squeeze(obs), self.X_GOAL[info['current_step']//20, :]))
            obs = torch.FloatTensor(extended_obs).to(self.device)
            action = self.agent.ac.act(obs)

        action = np.clip(np.squeeze([action]), [-1, -1], [1, 1])*0.25

        return action

    def run(self,
            n_episodes=10,
            ):
        '''Runs evaluation with current policy.'''
        self.agent.eval()
        obs, info = self.firmware_wrapper.reset()
        obs = np.squeeze(obs.reshape((12, 1))[:4, :])
        firmware_action = np.zeros((4,1))
        total_rew, total_violations, total_mse = 0, 0, 0
        ep_lengths, ep_returns, ep_violations, ep_mse = [], [], [], []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)

            # Safety filter
            certified_action, success = self.safety_filter.certify_action(obs, action, info)
            if success and self.filter_train_actions is True:
                action = certified_action
            else:
                self.safety_filter.ocp_solver.reset()
                certified_action, success = self.safety_filter.certify_action(obs, action, info)
                if success and self.filter_train_actions is True:
                    action = certified_action

            pos = [(action[0] + obs[0]), (action[1] + obs[2]), 1]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
            yaw = 0
            rpy_rate = [0, 0, 0]
            args = [pos, vel, acc, yaw, rpy_rate]
            curr_time = info['current_step']//20 * self.CTRL_DT
            self.firmware_wrapper.sendFullStateCmd(*args, curr_time)

            # Step the environment.
            next_obs, rew, done, info, firmware_action = self.firmware_wrapper.step(curr_time, firmware_action)
            next_obs = np.squeeze(next_obs.reshape((12, 1))[:4, :])
            total_violations += np.sum(np.abs(next_obs) > [0.75, 0.5, 0.75, 0.5])
            rew, mse = self.get_reward(next_obs, info)

            total_rew += rew
            total_mse += mse

            if done or info['current_step']//20-1 >= self.X_GOAL.shape[0] - 1 or self.firmware_wrapper._error == True:
                ep_lengths.append(info["current_step"]//20)
                ep_returns.append(total_rew)
                ep_violations.append(total_violations)
                ep_mse.append(total_mse/ep_lengths[-1])
                obs, info = self.env_reset(self.firmware_wrapper)
                obs = np.squeeze(obs.reshape((12, 1))[:4, :])
                firmware_action = np.zeros((4,1))
                total_rew, total_violations, total_mse = 0, 0, 0
                continue

            obs = next_obs

        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        ep_violations = np.asarray(ep_violations)
        ep_mse = np.asarray(ep_mse)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths, 'ep_violations': ep_violations, 'ep_mse': ep_mse}
        return eval_results

    def train_step(self):
        '''Performs a training/fine-tuning step.'''
        self.agent.train()
        rollouts = PPOBuffer(self.obs_space, self.act_space, self.rollout_steps, self.rollout_batch_size)
        obs, info = self.firmware_wrapper.reset()
        obs = np.squeeze(obs.reshape((12, 1))[:4, :])
        firmware_action = np.zeros((4,1))
        start = time.time()
        total_rew, total_violations, total_mse = 0, 0, 0
        ep_lengths, ep_returns, ep_violations, ep_mse = [], [], [], []
        for _ in range(self.rollout_steps):
            extended_obs = np.concatenate((obs, self.X_GOAL[info['current_step']//20, :]))
            with torch.no_grad():
                unsafe_action, v, logp = self.agent.ac.step(torch.FloatTensor(extended_obs).to(self.device))

            action = np.clip(np.squeeze([unsafe_action]), [-1, -1], [1, 1])*0.25
            scaled_unsafe_action = action

            # Adding safety filter
            success = False
            if self.safety_filter is not None and (self.filter_train_actions is True or self.penalize_sf_diff is True):
                certified_action, success = self.safety_filter.certify_action(obs, action, info)
                if success and self.filter_train_actions is True:
                    action = certified_action
                else:
                    self.safety_filter.ocp_solver.reset()
                    certified_action, success = self.safety_filter.certify_action(obs, action, info)
                    if success and self.filter_train_actions is True:
                        action = certified_action

            pos = [(action[0] + obs[0]), (action[1] + obs[2]), 1]
            vel = [0, 0, 0]
            acc = [0, 0, 0]
            yaw = 0
            rpy_rate = [0, 0, 0]
            args = [pos, vel, acc, yaw, rpy_rate]
            curr_time = info['current_step']//20 * self.CTRL_DT
            self.firmware_wrapper.sendFullStateCmd(*args, curr_time)

            # Step the environment.
            next_obs, rew, done, info, firmware_action = self.firmware_wrapper.step(curr_time, firmware_action)
            next_obs = np.squeeze(next_obs.reshape((12, 1))[:4, :])
            total_violations += np.sum(np.abs(next_obs) > [0.75, 0.5, 0.75, 0.5])
            rew, mse = self.get_reward(next_obs, info)

            if self.penalize_sf_diff and success:
                rew = np.log(rew)
                rew -= self.sf_penalty * np.linalg.norm(scaled_unsafe_action - certified_action)
                rew = np.exp(rew)

            # Constraint Penalty
            if self.firmware_wrapper.env.use_constraint_penalty and np.any(np.abs(next_obs) > [0.75, 0.5, 0.75, 0.5]):
                rew = np.log(rew)
                rew -= 1.0
                rew = np.exp(rew)

            total_rew += rew
            total_mse += mse

            if done or info['current_step']//20-1 >= self.X_GOAL.shape[0] - 1 or self.firmware_wrapper._error == True:
                ep_lengths.append(info["current_step"]//20)
                ep_returns.append(total_rew)
                ep_violations.append(total_violations)
                ep_mse.append(total_mse/ep_lengths[-1])
                total_rew, total_violations, total_mse = 0, 0, 0

                next_obs, info = self.env_reset(self.firmware_wrapper)
                next_obs = np.squeeze(next_obs.reshape((12, 1))[:4, :])
                firmware_action = np.zeros((4,1))
                mask = 0
            else:
                mask = 1

            # Time truncation is not the same as true termination.
            terminal_v = np.zeros_like(v)
            if 'terminal_info' in info:
                inff = info['terminal_info']
                if 'TimeLimit.truncated' in inff and inff['TimeLimit.truncated']:
                    terminal_obs = info['terminal_observation']
                    terminal_obs_tensor = torch.FloatTensor(terminal_obs).unsqueeze(0).to(self.device)
                    terminal_val = self.agent.ac.critic(terminal_obs_tensor).squeeze().detach().cpu().numpy()
                    terminal_v = terminal_val

            rollouts.push({'obs': extended_obs, 'act': unsafe_action, 'rew': rew, 'mask': mask, 'v': v, 'logp': logp, 'terminal_v': terminal_v})
            obs = next_obs

        self.total_steps += self.rollout_batch_size * self.rollout_steps
        # Learn from rollout batch.
        extended_obs = np.concatenate((obs, self.X_GOAL[info['current_step']//20, :]))
        last_val = self.agent.ac.critic(torch.FloatTensor(extended_obs).to(self.device)).detach().cpu().numpy()
        last_val = last_val.reshape((1,1))
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

        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        ep_violations = np.asarray(ep_violations)
        ep_mse = np.asarray(ep_mse)
        results.update({'ep_returns': ep_returns, 'ep_lengths': ep_lengths, 'ep_violations': ep_violations, 'ep_mse': ep_mse})
        return results

    def get_reward(self, obs, info):
        wp_idx = min(info['current_step']//20, self.X_GOAL.shape[0] - 1)  # +1 because state has already advanced but counter not incremented.
        state_error = obs - self.X_GOAL[wp_idx]
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
                for k in ['policy_loss', 'value_loss', 'entropy_loss', 'approx_kl']
            },
            step,
            prefix='loss')
        # Performance stats.
        ep_lengths = results['ep_lengths']
        ep_returns = results['ep_returns']
        ep_constraint_violation = results['ep_violations']
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

    def env_reset(self, env):
        '''Resets the environment until a feasible initial state is found.

        Args:
            env (BenchmarkEnv): The environment that is being reset.

        Returns:
            obs (ndarray): The initial observation.
            info (dict): The initial info.
        '''
        success = False
        act = np.array([0,0])
        obs, info = env.reset()
        if self.safety_filter is not None:
            self.safety_filter.reset_before_run()

        if self.use_safe_reset is True and self.safety_filter is not None:
            while success is not True or np.any(self.safety_filter.slack_prev > 1e-4):
                obs, info = env.reset()
                info['current_step'] = 1
                self.safety_filter.reset_before_run()
                _, success = self.safety_filter.certify_action(np.squeeze(obs.reshape((12, 1))[:4, :]), act, info)
                if not success:
                    self.safety_filter.ocp_solver.reset()
                    _, success = self.safety_filter.certify_action(np.squeeze(obs.reshape((12, 1))[:4, :]), act, info)

        return obs, info
