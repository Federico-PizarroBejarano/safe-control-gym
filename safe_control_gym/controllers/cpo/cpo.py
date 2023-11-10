'''Constrained Policy Optimization (CPO)

Based on:
    * https://github.com/dobro12/CPO

Additional references:
    * Constrained Policy Optimization - https://arxiv.org/abs/1705.10528
    * Original Implementation - https://github.com/jachiam/cpo
    * safety-starter-agents (CPO) - https://github.com/openai/safety-starter-agents
'''

import os
import time
from copy import deepcopy

import numpy as np
import torch

from safe_control_gym.controllers.base_controller import BaseController
from safe_control_gym.controllers.cpo.cpo_utils import CPOPolicy, CPOValue
from safe_control_gym.envs.env_wrappers.record_episode_statistics import RecordEpisodeStatistics
from safe_control_gym.utils.logging import ExperimentLogger

EPS = 1e-8


@torch.jit.script
def normalize(a, maximum, minimum):
    temp_a = 1.0 / (maximum - minimum)
    temp_b = minimum / (minimum - maximum)
    temp_a = torch.ones_like(a) * temp_a
    temp_b = torch.ones_like(a) * temp_b
    return temp_a * a + temp_b


@torch.jit.script
def unnormalize(a, maximum, minimum):
    temp_a = maximum - minimum
    temp_b = minimum
    temp_a = torch.ones_like(a) * temp_a
    temp_b = torch.ones_like(a) * temp_b
    return temp_a * a + temp_b


@torch.jit.script
def clip(a, maximum, minimum):
    clipped = torch.where(a > maximum, maximum, a)
    clipped = torch.where(clipped < minimum, minimum, clipped)
    return clipped


def flatGrad(y, x, retain_graph=False, create_graph=False):
    if create_graph:
        retain_graph = True
    g = torch.autograd.grad(y, x, retain_graph=retain_graph, create_graph=create_graph)
    g = torch.cat([t.view(-1) for t in g])
    return g


class CPO(BaseController):
    '''Constrained policy optimization.'''

    def __init__(self,
                 env_func,
                 training=True,
                 checkpoint_path='model_latest.pt',
                 output_dir='temp',
                 use_gpu=False,
                 seed=0,
                 **kwargs):
        super().__init__(env_func, training, checkpoint_path, output_dir, use_gpu, seed, **kwargs)
        # Task.
        if self.training:
            # Training and testing.
            self.env = env_func(seed=seed)
            self.env = RecordEpisodeStatistics(self.env, 10)
            self.eval_env = env_func(seed=seed * 111)
            self.eval_env = RecordEpisodeStatistics(self.eval_env, 10)
        else:
            # Testing only.
            self.env = env_func()
            self.env = RecordEpisodeStatistics(self.env)

        # Logging.
        if self.training:
            log_file_out = True
            use_tensorboard = self.tensorboard
        else:
            # Disable logging to file and tfboard for evaluation.
            log_file_out = False
            use_tensorboard = False
        self.logger = ExperimentLogger(output_dir, log_file_out=log_file_out, use_tensorboard=use_tensorboard)

        # constant about env
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound_min = torch.tensor(self.env.action_space.low, device=self.device)
        self.action_bound_max = torch.tensor(self.env.action_space.high, device=self.device)

        # declare value and policy
        self.policy = CPOPolicy(self.obs_dim, self.action_dim, self.hidden1, self.hidden2).to(self.device)
        self.value = CPOValue(self.obs_dim, self.hidden1, self.hidden2).to(self.device)
        self.cost_value = CPOValue(self.obs_dim, self.hidden1, self.hidden2).to(self.device)
        self.v_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.v_lr)
        self.cost_v_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=self.cost_v_lr)

        self.policy.initialize()
        self.value.initialize()
        self.cost_value.initialize()

        self.epoch = 0
        self.reset()

    def save(self, path):
        '''Saves model params and experiment state to checkpoint path.'''
        torch.save({
            'epoch': self.epoch,
            'policy': self.policy.state_dict(),
            'value': self.value.state_dict(),
            'cost_value': self.cost_value.state_dict(),
            'v_optimizer': self.v_optimizer.state_dict(),
            'cost_v_optimizer': self.cost_v_optimizer.state_dict(),
        }, path)

    def load(self, path):
        '''Restores model and experiment given checkpoint path.'''
        checkpoint = torch.load(path)
        self.epoch = checkpoint['epoch'] + 1
        self.policy.load_state_dict(checkpoint['policy'])
        self.value.load_state_dict(checkpoint['value'])
        self.cost_value.load_state_dict(checkpoint['cost_value'])
        self.v_optimizer.load_state_dict(checkpoint['v_optimizer'])
        self.cost_v_optimizer.load_state_dict(checkpoint['cost_v_optimizer'])

    def reset(self):
        '''Do initializations for training or evaluation.'''
        if self.training:
            # set up stats tracking
            self.env.add_tracker('constraint_violation', 0)
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('constraint_violation', 0, mode='queue')
            self.eval_env.add_tracker('mse', 0, mode='queue')

            self.env.reset()
            self.eval_env.reset()
        else:
            # Add episodic stats to be tracked.
            self.env.add_tracker('constraint_violation', 0, mode='queue')
            self.env.add_tracker('constraint_values', 0, mode='queue')
            self.env.add_tracker('mse', 0, mode='queue')
            self.env.reset()

    def close(self):
        '''Shuts down and cleans up lingering resources.'''
        self.env.close()
        if self.training:
            self.eval_env.close()
        self.logger.close()

    def select_action(self, obs, info=None, is_train=False):
        '''
        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.
            is_train (bool): Whether the action is being called during training.
        Returns:
            action (ndarray): The action chosen by the controller.
            clipped_action (ndarray): The clipped action.
        '''
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            mean, _, std = self.policy(obs)

        if is_train:
            noise = torch.randn(*mean.size(), device=self.device)
            action = self.unnormalizeAction(mean + noise * std)
        else:
            action = self.unnormalizeAction(mean)
        clipped_action = clip(action, self.action_bound_max, self.action_bound_min)

        if is_train:
            return action, clipped_action
        else:
            return clipped_action.detach().cpu().numpy()

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Performs learning (pre-training, training, fine-tuning, etc).'''
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            results = self.train_loop(epoch)
            # Evaluation.
            if self.eval_interval and (epoch + 1) % self.eval_interval == 0:
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
            if self.log_interval and (epoch + 1) % self.log_interval == 0:
                self.log_step(results)

        # Latest/final checkpoint.
        self.save(self.checkpoint_path)
        self.logger.info(f'Checkpoint | {self.checkpoint_path}')

    def train_loop(self, epoch):
        trajectories = []
        ep_step = 0
        scores = []
        cvs = []
        start = time.time()
        while ep_step <= self.max_steps:
            obs, info = self.env.reset()
            score = 0
            cv = 0
            done = False
            while not done:
                ep_step += 1
                obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float)
                action_tensor, clipped_action_tensor = self.select_action(obs_tensor, info, is_train=True)
                action = action_tensor.detach().cpu().numpy()
                clipped_action = clipped_action_tensor.detach().cpu().numpy()
                next_obs, reward, done, info = self.env.step(clipped_action)
                cost = self.get_cost(info)

                fail = False
                if done:
                    if 'TimeLimit.truncated' in info:
                        fail = not info['TimeLimit.truncated']
                    else:
                        fail = True

                trajectories.append([obs, action, reward, cost, done, fail, next_obs])

                obs = next_obs
                score += reward
                cv += np.sum(info['constraint_values'] >= 0)

            scores.append(score)
            cvs.append(cv)

        v_loss, cost_v_loss, objective, cost_surrogate, kl, entropy = self.train_step(trajs=trajectories)
        score = np.mean(scores)
        cvs = np.mean(cvs)
        results = {'score': score, 'cv': cv, 'value_loss': v_loss, 'cost_value_loss': cost_v_loss, 'objective': objective, 'cost_surrogate': cost_surrogate, 'approx_kl': kl, 'entropy_loss': entropy}
        results.update({'step': (epoch + 1) * self.max_steps, 'elapsed_time': time.time() - start})
        return results

    def train_step(self, trajs):
        '''Performs a training/fine-tuning step.'''
        # convert to numpy array
        obs = np.array([traj[0] for traj in trajs])
        actions = np.array([traj[1] for traj in trajs])
        rewards = np.array([traj[2] for traj in trajs])
        costs = np.array([traj[3] for traj in trajs])
        dones = np.array([traj[4] for traj in trajs])
        fails = np.array([traj[5] for traj in trajs])
        next_obs = np.array([traj[6] for traj in trajs])

        # convert to tensor
        obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float)
        actions_tensor = torch.tensor(actions, device=self.device, dtype=torch.float)
        norm_actions_tensor = self.normalizeAction(actions_tensor)
        next_obs_tensor = torch.tensor(next_obs, device=self.device, dtype=torch.float)

        # get GAEs and Tagets
        # for reward
        values_tensor = self.value(obs_tensor)
        next_values_tensor = self.value(next_obs_tensor)
        values = values_tensor.detach().cpu().numpy()
        next_values = next_values_tensor.detach().cpu().numpy()
        gaes, targets = self.getGaesTargets(rewards, values, dones, fails, next_values)
        gaes_tensor = torch.tensor(gaes, device=self.device, dtype=torch.float)
        targets_tensor = torch.tensor(targets, device=self.device, dtype=torch.float)
        # for cost
        cost_values_tensor = self.cost_value(obs_tensor)
        next_cost_values_tensor = self.cost_value(next_obs_tensor)
        cost_values = cost_values_tensor.detach().cpu().numpy()
        next_cost_values = next_cost_values_tensor.detach().cpu().numpy()
        cost_gaes, cost_targets = self.getGaesTargets(costs, cost_values, dones, fails, next_cost_values)
        cost_gaes_tensor = torch.tensor(cost_gaes, device=self.device, dtype=torch.float)
        cost_targets_tensor = torch.tensor(cost_targets, device=self.device, dtype=torch.float)

        # get cost mean
        cost_mean = np.mean(costs) / (1 - self.discount_factor)

        # get entropy
        entropy = self.getEntropy(obs_tensor)

        # ======================================= #
        # ========== for policy update ========== #
        # backup old policy
        means, _, stds = self.policy(obs_tensor)
        old_means = means.clone().detach()
        old_stds = stds.clone().detach()

        # get objective & KL & cost surrogate
        objective = self.getObjective(obs_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
        cost_surrogate = self.getCostSurrogate(obs_tensor, norm_actions_tensor, old_means, old_stds, cost_gaes_tensor, cost_mean)
        kl = self.getKL(obs_tensor, old_means, old_stds)

        # get gradient
        grad_g = flatGrad(objective, self.policy.parameters(), retain_graph=True)
        grad_b = flatGrad(-cost_surrogate, self.policy.parameters(), retain_graph=True)
        x_value = self.conjugateGradient(kl, grad_g)
        approx_g = self.Hx(kl, x_value)
        cost_d = self.cost_d / (1.0 - self.discount_factor)
        c_value = cost_surrogate - cost_d

        # solve Lagrangian problem
        if torch.dot(grad_b, grad_b) <= 1e-8 and c_value < 0:
            H_inv_b, scalar_r, scalar_s, A_value, B_value = 0, 0, 0, 0, 0
            scalar_q = torch.dot(approx_g, x_value)
            optim_case = 4
        else:
            H_inv_b = self.conjugateGradient(kl, grad_b)
            approx_b = self.Hx(kl, H_inv_b)
            scalar_q = torch.dot(approx_g, x_value)
            scalar_r = torch.dot(approx_g, H_inv_b)
            scalar_s = torch.dot(approx_b, H_inv_b)
            A_value = scalar_q - scalar_r**2 / scalar_s  # should be always positive (Cauchy-Shwarz)
            B_value = 2 * self.max_kl - c_value**2 / scalar_s  # does safety boundary intersect trust region? (positive = yes)
            if c_value < 0 and B_value < 0:
                optim_case = 3
            elif c_value < 0 and B_value >= 0:
                optim_case = 2
            elif c_value >= 0 and B_value >= 0:
                optim_case = 1
            else:
                optim_case = 0
        # print('optimizing case :', optim_case)
        if optim_case in [3, 4]:
            lam = torch.sqrt(scalar_q / (2 * self.max_kl))
            nu = 0
        elif optim_case in [1, 2]:
            def proj(x, L):
                return max(L[0], min(L[1], x))

            def f_a(lam):
                return -0.5 * (A_value / (lam + EPS) + B_value * lam) - scalar_r * c_value / (scalar_s + EPS)

            def f_b(lam):
                return -0.5 * (scalar_q / (lam + EPS) + 2 * self.max_kl * lam)

            LA, LB = [0, scalar_r / c_value], [scalar_r / c_value, np.inf]
            LA, LB = (LA, LB) if c_value < 0 else (LB, LA)

            lam_a = proj(torch.sqrt(A_value / B_value), LA)
            lam_b = proj(torch.sqrt(scalar_q / (2 * self.max_kl)), LB)
            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0, lam * c_value - scalar_r) / (scalar_s + EPS)
        else:
            lam = 0
            nu = torch.sqrt(2 * self.max_kl / (scalar_s + EPS))

        # line search
        delta_theta = (1. / (lam + EPS)) * (x_value + nu * H_inv_b) if optim_case > 0 else nu * H_inv_b
        beta = 1.0
        init_theta = torch.cat([t.view(-1) for t in self.policy.parameters()]).clone().detach()
        init_objective = objective.clone().detach()
        init_cost_surrogate = cost_surrogate.clone().detach()
        while True:
            theta = beta * delta_theta + init_theta
            self.applyParams(theta)
            objective = self.getObjective(obs_tensor, norm_actions_tensor, gaes_tensor, old_means, old_stds)
            cost_surrogate = self.getCostSurrogate(obs_tensor, norm_actions_tensor, old_means, old_stds, cost_gaes_tensor, cost_mean)
            kl = self.getKL(obs_tensor, old_means, old_stds)
            if kl <= self.max_kl and (objective > init_objective if optim_case > 1 else True) and cost_surrogate - init_cost_surrogate <= max(-c_value, 0):
                break
            beta *= self.line_decay
        # ======================================= #

        # ======================================== #
        # =========== for value update =========== #
        for _ in range(self.value_epochs):
            value_loss = torch.mean(0.5 * torch.square(self.value(obs_tensor) - targets_tensor))
            self.v_optimizer.zero_grad()
            value_loss.backward()
            self.v_optimizer.step()

            cost_value_loss = torch.mean(0.5 * torch.square(self.cost_value(obs_tensor) - cost_targets_tensor))
            self.cost_v_optimizer.zero_grad()
            cost_value_loss.backward()
            self.cost_v_optimizer.step()
        # ======================================== #

        def scalar(x):
            return x.detach().cpu().numpy()

        np_value_loss = scalar(value_loss)
        np_cost_value_loss = scalar(cost_value_loss)
        np_objective = scalar(objective)
        np_cost_surrogate = scalar(cost_surrogate)
        np_kl = scalar(kl)
        np_entropy = scalar(entropy)
        return np_value_loss, np_cost_value_loss, np_objective, np_cost_surrogate, np_kl, np_entropy

    def run(self, n_episodes=10):
        '''Runs evaluation with current policy.'''
        obs, info = self.eval_env.reset()
        ep_returns, ep_lengths = [], []
        while len(ep_returns) < n_episodes:
            action = self.select_action(obs=obs, info=info)
            obs, _, done, info = self.eval_env.step(action)
            if done:
                assert 'episode' in info
                ep_returns.append(info['episode']['r'])
                ep_lengths.append(info['episode']['l'])
                obs, _ = self.eval_env.reset()
        # Collect evaluation results.
        ep_lengths = np.asarray(ep_lengths)
        ep_returns = np.asarray(ep_returns)
        eval_results = {'ep_returns': ep_returns, 'ep_lengths': ep_lengths}
        # Other episodic stats from evaluation env.
        if len(self.eval_env.queued_stats) > 0:
            queued_stats = {k: np.asarray(v) for k, v in self.eval_env.queued_stats.items()}
            eval_results.update(queued_stats)
        return eval_results

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
                'progress': step / self.max_steps / self.num_epochs
            },
            step,
            prefix='time',
            write=False,
            write_tb=False)
        # Learning stats.
        self.logger.add_scalars(
            {
                k: results[k]
                for k in ['value_loss', 'cost_value_loss', 'objective', 'cost_surrogate', 'approx_kl', 'entropy_loss']
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
                'ep_constraint_violation': ep_constraint_violation.mean()
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
                    'mse': eval_mse.mean()
                },
                step,
                prefix='stat_eval')
        # Print summary table
        self.logger.dump_scalars()

    def get_cost(self, info):
        '''Calculates the cost surrogate function for the current constraints.

        Args:
            info (dict): The info at the current timestep.

        Returns:
            constraint_cost (float): The cost at the current constraint values.
        '''
        nx = self.env.symbolic.nx

        state_constraints = np.maximum(info['constraint_values'][:nx], info['constraint_values'][nx:nx * 2])
        constraint_width = info['constraint_values'][:nx] + info['constraint_values'][nx:nx * 2]
        state_cost = np.divide(state_constraints, -constraint_width / 2) + 0.0001

        if np.any(state_cost >= 0):
            return np.max(state_cost)
        else:
            return 0

    def normalizeAction(self, a):
        return normalize(a, self.action_bound_max, self.action_bound_min)

    def unnormalizeAction(self, a):
        return unnormalize(a, self.action_bound_max, self.action_bound_min)

    def getGaesTargets(self, rewards, values, dones, fails, next_values):
        '''
        Args:
            rewards (ndarray[n_steps,]):
            values (ndarray[n_steps,]):
            dones (ndarray[n_steps,]):
            fails (ndarray[n_steps,]):
            next_values (ndarray[n_steps,]):
        Returns:
            gaes (ndarray[n_steps,]):
            targets (ndarray[n_steps,]):
        '''
        deltas = rewards + (1.0 - fails) * self.discount_factor * next_values - values
        gaes = deepcopy(deltas)
        for t in reversed(range(len(gaes))):
            if t < len(gaes) - 1:
                gaes[t] = gaes[t] + (1.0 - dones[t]) * self.discount_factor * self.gae_coeff * gaes[t + 1]
        targets = values + gaes
        return gaes, targets

    def getEntropy(self, obs):
        '''
        Return a scalar tensor for the entropy value.

        Args:
            obs (Tensor[n_steps, obs_dim]):
        Returns:
            entropy (Tensor[]):
        '''
        means, _, stds = self.policy(obs)
        normal = torch.distributions.Normal(means, stds)
        entropy = torch.mean(torch.sum(normal.entropy(), dim=1))
        return entropy

    def getObjective(self, obs, norm_actions, gaes, old_means, old_stds):
        means, _, stds = self.policy(obs)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)
        objective = torch.mean(torch.exp(log_probs - old_log_probs) * gaes)
        return objective

    def getCostSurrogate(self, obs, norm_actions, old_means, old_stds, cost_gaes, cost_mean):
        means, _, stds = self.policy(obs)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        log_probs = torch.sum(dist.log_prob(norm_actions), dim=1)
        old_log_probs = torch.sum(old_dist.log_prob(norm_actions), dim=1)
        cost_surrogate = cost_mean + (1.0 / (1.0 - self.discount_factor)) * (torch.mean(torch.exp(log_probs - old_log_probs) * cost_gaes) - torch.mean(cost_gaes))
        return cost_surrogate

    def getKL(self, obs, old_means, old_stds):
        means, _, stds = self.policy(obs)
        dist = torch.distributions.Normal(means, stds)
        old_dist = torch.distributions.Normal(old_means, old_stds)
        kl = torch.distributions.kl.kl_divergence(old_dist, dist)
        kl = torch.mean(torch.sum(kl, dim=1))
        return kl

    def applyParams(self, params):
        n = 0
        for p in self.policy.parameters():
            numel = p.numel()
            g = params[n:n + numel].view(p.shape)
            p.data = g
            n += numel

    def Hx(self, kl, x):
        '''
        Get Hx (Hessian of KL * x).
         Args:
            kl (Tensor[]):
            x (Tensor[dim,]):
        Returns:
            Hx (Tensor[dim,]):
        '''
        flat_grad_kl = flatGrad(kl, self.policy.parameters(), create_graph=True)
        kl_x = torch.dot(flat_grad_kl, x)
        H_x = flatGrad(kl_x, self.policy.parameters(), retain_graph=True)
        return H_x + x * self.damping_coeff

    def conjugateGradient(self, kl, g):
        '''
        Get (H^{-1} * g).

        Args:
            kl (Tensor[]):
            g (Tensor[dim,]):
        Returns:
            H^{-1}g (Tensor[dim,]):
        '''
        x = torch.zeros_like(g, device=self.device)
        r = g.clone()
        p = g.clone()
        rs_old = torch.sum(r * r)
        for _ in range(self.num_conjugate):
            Ap = self.Hx(kl, p)
            pAp = torch.sum(p * Ap)
            alpha = rs_old / (pAp + EPS)
            x += alpha * p
            r -= alpha * Ap
            rs_new = torch.sum(r * r)
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x
