'''Precomputed Cost Function for Smooth MPSC. '''

import numpy as np

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST
from safe_control_gym.controllers.ppo.ppo import PPO
from safe_control_gym.controllers.sac.sac import SAC
from safe_control_gym.envs.benchmark_env import Environment


class PRECOMPUTED_COST(MPSC_COST):
    '''Precomputed future states MPSC Cost Function. '''

    def __init__(self,
                 env,
                 horizon: int = 10,
                 output_dir: str = '.',
                 **kwargs
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env (BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            output_dir (str): Folder to write outputs.
        '''

        self.env = env
        self.model = self.env.symbolic

        self.horizon = horizon
        self.output_dir = output_dir

        if 'decay_factor' in kwargs:
            self.decay_factor = kwargs['decay_factor']
        else:
            self.decay_factor = 1

        self.uncertified_controller = None

    def get_cost(self, opti_dict):
        '''Returns the cost function for the MPSC optimization in symbolic form.

        Args:
            opti_dict (dict): The dictionary of optimization variables.

        Returns:
            cost (casadi symbolic expression): The symbolic cost function using casadi.
        '''

        opti = opti_dict['opti']
        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']
        v_var = opti_dict['v_var']

        nu = self.model.nu

        v_L = opti.parameter(nu, self.horizon)

        opti_dict['v_L'] = v_L

        cost = (u_L - next_u).T @ (u_L - next_u)
        for h in range(1, self.horizon):
            cost += (self.decay_factor**h)*(v_L[:, h] - v_var[:, h]).T @ (v_L[:, h] - v_var[:, h])

        return cost

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepares all the symbolic variable initial values for the next optimization.

        Args:
            opti_dict (dict): The dictionary of optimization variables.
            obs (ndarray): Current state/observation.
            iteration (int): The current iteration, used for trajectory tracking.
        '''

        opti = opti_dict['opti']
        v_L = opti_dict['v_L']
        u_L = opti_dict['u_L']

        uncertified_action = opti.value(u_L, opti.initial())

        expected_inputs = self.calculate_unsafe_path(obs, uncertified_action, iteration)
        opti.set_value(v_L, expected_inputs)

    def calculate_unsafe_path(self, obs, uncertified_action, iteration):
        '''Precomputes the likely actions the uncertified controller will take.

        Args:
            obs (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            v_L (ndarray): The estimated future actions taken by the uncertified_controller.
        '''

        if self.uncertified_controller is None:
            raise Exception('[ERROR] No underlying controller passed to P_MPSC')

        v_L = np.zeros((self.model.nu, self.horizon))

        if isinstance(self.uncertified_controller, (PPO, SAC)):
            file_end = 'pt'
        else:
            file_end = 'npy'

        self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_curr.{file_end}')
        self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_prev.{file_end}')

        for h in range(self.horizon):
            next_step = min(iteration+h, self.env.X_GOAL.shape[0]-1)
            # Concatenate goal info (goal state(s)) for RL
            if next_step == 0:
                extended_obs = self.env.extend_obs(obs, 1)
            else:
                extended_obs = self.env.extend_obs(obs, next_step)

            action = self.uncertified_controller.select_action(obs=extended_obs, info={'current_step': next_step})

            if self.uncertified_controller.env.NORMALIZED_RL_ACTION_SPACE:
                if self.env.NAME == Environment.CARTPOLE:
                    action = self.uncertified_controller.env.action_scale * action
                elif self.env.NAME == Environment.QUADROTOR:
                    action = (1 + self.uncertified_controller.env.norm_act_scale * action) * self.uncertified_controller.env.hover_thrust

            if h == 0 and np.linalg.norm(uncertified_action - action) >= 0.001:
                print(f'MISMATCH BETWEEN Unsafe Controller AND MPSC Guess!!!! Uncert: {uncertified_action}, Guess: {action}, Diff: {np.linalg.norm(uncertified_action - action)}')
                raise ValueError()

            v_L[:, h:h+1] = action.reshape((self.model.nu, 1))
            action = np.clip(action, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])

            obs = np.squeeze(self.model.fd_func(x0=obs, p=action)['xf'].toarray())

        self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_curr.{file_end}')
        self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_prev.{file_end}')

        return v_L
