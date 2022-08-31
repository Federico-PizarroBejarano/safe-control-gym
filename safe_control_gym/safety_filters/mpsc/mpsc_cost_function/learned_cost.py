'''Learned Cost Function for Smooth MPSC. '''

import pickle
from itertools import product

import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST
from safe_control_gym.envs.benchmark_env import Task


class LEARNED_COST(MPSC_COST):
    '''Learned future states MPSC Cost Function. '''

    def __init__(self,
                 env,
                 horizon: int = 10,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env (BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
        '''

        self.env = env
        self.model = self.env.symbolic

        self.horizon = horizon

        self.mpsc_cost_horizon = mpsc_cost_horizon
        self.decay_factor = decay_factor

        self.uncertified_controller = None

        self.max_order = 2
        self.power_list = [p for p in product(range(self.max_order+1), repeat=self.model.nx) if sum(p) <= self.max_order]

        self.gamma = self.transform_state_errors(np.zeros((self.model.nx)))
        self.prev_actions = np.zeros((1, self.model.nu))
        self.generate_policy()

    def get_cost(self, opti_dict):
        '''Returns the cost function for the MPSC optimization in symbolic form.

        Args:
            opti_dict (dict): The dictionary of optimization variables.

        Returns:
            cost (casadi symbolic expression): The symbolic cost function using casadi.
        '''

        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']
        z_var = opti_dict['z_var']
        v_var = opti_dict['v_var']
        X_GOAL = opti_dict['X_GOAL']

        nx = self.model.nx

        if 'X_EQ' in opti_dict:
            X_EQ = opti_dict['X_EQ']
        else:
            X_EQ = cs.MX(np.zeros(nx))

        cost = (u_L - next_u).T @ (u_L - next_u)
        for h in range(1, self.mpsc_cost_horizon):
            if self.env.TASK == Task.STABILIZATION:
                state_error = z_var[:, h] - X_GOAL.T + X_EQ
            elif self.env.TASK == Task.TRAJ_TRACKING:
                state_error = z_var[:, h] - X_GOAL[h, :].T + X_EQ
            v_L = self.policy(state_error) + self.env.U_EQ
            cost += (self.decay_factor**h)*(v_L - v_var[:, h]).T @ (v_L - v_var[:, h])

        return cost

    def learn_policy(self, path):
        '''Attempts to learn a model of the uncertified_controllers policy.

        Args:
            path (str): The path to the past trajectory information.
        '''

        with open(path, 'rb') as file:
            trajs_data = pickle.load(file)

        for ep in range(len(trajs_data['action'])):
            for ite in range(trajs_data['action'][ep].shape[0]):
                obs, action = trajs_data['state'][ep][ite], trajs_data['current_physical_action'][ep][ite]
                self.update_policy(obs[0:self.model.nx], action, ite)

        self.generate_policy()

    def update_policy(self, obs, u_L, iteration=0):
        '''
        Updates the learned policy.

        Args:
            obs (ndarray): Current state/observation.
            u_L (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.
        '''

        if self.env.TASK == Task.STABILIZATION:
            state_error = np.squeeze(obs) - np.squeeze(self.env.X_GOAL)
        else:
            state_error = np.squeeze(obs) - np.squeeze(np.atleast_2d(self.env.X_GOAL)[iteration, :])

        self.prev_actions = np.vstack((self.prev_actions, u_L-self.env.U_EQ))
        transformed_error = self.transform_state_errors(state_error)
        self.gamma = np.vstack((self.gamma, transformed_error))

    def transform_state_errors(self, state_error):
        '''
        Transforms the state error into the form used by linear regression.

        Args:
            state error (ndarray): The state error (current state - current goal).

        Returns:
            transformed_error (ndarray): The transformed state error.
        '''

        gamma = []

        for powers in self.power_list:
            monomial = 1.0
            for state in range(self.model.nx):
                monomial *= (state_error[state]**powers[state])
            gamma.append(monomial)

        return np.array([gamma])

    def generate_policy(self):
        '''Generates the symbolic policy based on the current collected data points. '''

        action = np.zeros((self.model.nu, 1))
        counter = 0

        self.weights = np.linalg.pinv(self.gamma.T @ self.gamma) @ self.gamma.T @ self.prev_actions
        state_error = cs.MX.sym('x', self.model.nx, 1)

        for powers in self.power_list:
            monomial = 1.0
            for state in range(self.model.nx):
                monomial *= (state_error[state]**powers[state])
            action += monomial*self.weights[counter]
            counter += 1

        self.policy = cs.Function('policy',
                [state_error],
                [action],
                ['state_error'],
                ['action'])
