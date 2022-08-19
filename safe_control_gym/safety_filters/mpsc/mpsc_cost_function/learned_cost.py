'''Learned Cost Function for Smooth MPSC'''

import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST
from safe_control_gym.controllers.ppo.ppo import PPO
from safe_control_gym.controllers.sac.sac import SAC
from safe_control_gym.envs.benchmark_env import Task, Environment


class LEARNED_COST(MPSC_COST):
    '''Learned future states MPSC Cost Function. '''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 output_dir: str = '.',
                 **kwargs
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            output_dir (str): Folder to write outputs.
        '''

        self.env = env_func()
        self.model = self.env.symbolic

        self.X_EQ = np.zeros(self.model.nx)
        self.U_EQ = np.atleast_2d(self.env.U_GOAL)[0, :]

        self.horizon = horizon
        self.output_dir = output_dir

        if 'decay_factor' in kwargs:
            self.decay_factor = kwargs['decay_factor']
        else:
            self.decay_factor = 1

        if 'num_policy_samples' in kwargs:
            self.num_policy_samples = kwargs['num_policy_samples']
        else:
            self.num_policy_samples = 10

        self.uncertified_controller = None

        self.prev_state_errors = np.zeros((1, self.model.nx))
        self.gamma = self.transform_state_errors(self.prev_state_errors)
        self.prev_actions = np.zeros((1, self.model.nu))
        self.generate_policy()

    def set_uncertified_controller(self, uncertified_controller):
        '''Sets the uncertified controller to be used.

        Args:
            uncertified_controller (BaseController): The uncertified controller to be certified.
        '''
        self.uncertified_controller = uncertified_controller

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
        for h in range(1, self.horizon):
            if self.env.TASK == Task.STABILIZATION:
                state_error = z_var[:, h] - X_GOAL.T + X_EQ
            elif self.env.TASK == Task.TRAJ_TRACKING:
                state_error = z_var[:, h] - X_GOAL[h, :].T + X_EQ
            v_L = self.policy(state_error)
            cost += (self.decay_factor**h)*(v_L - v_var[:, h]).T @ (v_L - v_var[:, h])

        return cost

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepares all the symbolic variable initial values for the next optimization.

        Args:
            opti_dict (dict): The dictionary of optimization variables.
            obs (ndarray): Current state/observation.
            iteration (int): The current iteration, used for trajectory tracking.
        '''

        opti = opti_dict['opti']
        u_L = opti_dict['u_L']

        uncertified_action = opti.value(u_L, opti.initial())

        self.update_policy(obs, uncertified_action, iteration)

    def learn_policy(self, init_state):
        '''Attempts to learn a model of the uncertified_controllers policy.

        Args:
            init_state (ndarray): The initial state to begin learning.
        '''

        if isinstance(self.uncertified_controller, (PPO, SAC)):
            file_end = 'pt'
        else:
            file_end = 'npy'

        self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_curr.{file_end}')
        self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_prev.{file_end}')

        if self.env.TASK == Task.STABILIZATION:
            training_points = np.linspace(init_state, self.env.X_GOAL, self.num_policy_samples*20)
            for iteration in range(training_points.shape[0]):
                test_obs = training_points[iteration, :] + np.random.rand((self.model.nx))/10 - 1/20
                extended_obs = self.env.extend_obs(obs=test_obs, next_step=1)
                action = self.uncertified_controller.select_action(obs=extended_obs, info={'current_step': iteration})
                self.update_policy(test_obs, action)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            for _ in range(self.num_policy_samples):
                for iteration in range(self.env.X_GOAL.shape[0]):
                    test_obs = self.env.X_GOAL[iteration, :] + np.random.rand((self.model.nx))/10 - 1/20
                    extended_obs = self.env.extend_obs(obs=test_obs, next_step=iteration)
                    if iteration == 0:
                        extended_obs = self.env.extend_obs(obs=test_obs, next_step=1)
                    else:
                        extended_obs = self.env.extend_obs(obs=test_obs, next_step=iteration)
                    action = self.uncertified_controller.select_action(obs=extended_obs, info={'current_step': iteration})
                    self.update_policy(test_obs, action, iteration)

        self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_curr.{file_end}')
        self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_prev.{file_end}')

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
            state_error = obs - self.env.X_GOAL
        else:
            state_error = obs - np.atleast_2d(self.env.X_GOAL)[iteration, :].T

        self.prev_state_errors = np.vstack((self.prev_state_errors, state_error))
        self.prev_actions = np.vstack((self.prev_actions, u_L-self.U_EQ))
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
        max_order = 2

        if self.env.NAME == Environment.CARTPOLE:
            x, v, theta, omega = np.squeeze(state_error)
            for x_order in range(0, max_order + 1):
                for v_order in range(0, max_order + 1):
                    for theta_order in range(0, max_order + 1):
                        for omega_order in range(0, max_order + 1):
                            if x_order + v_order + theta_order + omega_order > 2:
                                continue
                            monomial = (x**x_order) * (v**v_order) * (theta**theta_order) * (omega**omega_order)
                            gamma.append(monomial)
        elif self.env.NAME == Environment.QUADROTOR:
            x, v_x, z, v_z, theta, omega = np.squeeze(state_error)
            for x_order in range(0, max_order + 1):
                for vx_order in range(0, max_order + 1):
                    for z_order in range(0, max_order + 1):
                        for vz_order in range(0, max_order + 1):
                            for theta_order in range(0, max_order + 1):
                                for omega_order in range(0, max_order + 1):
                                    if x_order + vx_order + z_order + vz_order + theta_order + omega_order > 2:
                                        continue
                                    monomial = (x**x_order) * (v_x**vx_order) * (z**z_order) * (v_z**vz_order) * (theta**theta_order) * (omega**omega_order)
                                    gamma.append(monomial)

        return np.array([gamma])

    def generate_policy(self):
        '''Generates the symbolic policy based on the current collected data points. '''

        action = 0
        counter = 0
        max_order = 2

        self.weights = np.linalg.pinv(self.gamma.T @ self.gamma) @ self.gamma.T @ self.prev_actions

        if self.env.NAME == Environment.CARTPOLE:
            state_error = cs.MX.sym('x',4,1)
            for x_order in range(0, max_order + 1):
                for v_order in range(0, max_order + 1):
                    for theta_order in range(0, max_order + 1):
                        for omega_order in range(0, max_order + 1):
                            if x_order + v_order + theta_order + omega_order > 2:
                                continue
                            monomial = (state_error[0]**x_order) * (state_error[1]**v_order) * (state_error[2]**theta_order) * (state_error[3]**omega_order)
                            action += monomial*self.weights[counter]
                            counter += 1
        elif self.env.NAME == Environment.QUADROTOR:
            state_error = cs.MX.sym('x',6,1)
            for x_order in range(0, max_order + 1):
                for vx_order in range(0, max_order + 1):
                    for z_order in range(0, max_order + 1):
                        for vz_order in range(0, max_order + 1):
                            for theta_order in range(0, max_order + 1):
                                for omega_order in range(0, max_order + 1):
                                    if x_order + vx_order + z_order + vz_order + theta_order + omega_order > 2:
                                        continue
                                    monomial = (state_error[0]**x_order) * (state_error[1]**vx_order) * (state_error[2]**z_order) * (state_error[3]**vz_order) * (state_error[4]**theta_order) * (state_error[5]**omega_order)
                                    action += monomial*self.weights[counter]
                                    counter += 1

        self.policy = cs.Function('policy',
                [state_error],
                [action],
                ['state_error'],
                ['action'])
