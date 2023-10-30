'''Precomputed Cost Function for Smooth MPSC.'''

import numpy as np

from safe_control_gym.controllers.pid.pid import PID
from safe_control_gym.envs.benchmark_env import Environment
from safe_control_gym.envs.env_wrappers.vectorized_env.vec_env import VecEnv
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST


class PRECOMPUTED_COST(MPSC_COST):
    '''Precomputed future states MPSC Cost Function.'''

    def __init__(self,
                 env,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 output_dir: str = '.',
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env (BenchmarkEnv): Environment for the task.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
            output_dir (str): Folder to write outputs.
        '''

        super().__init__(env, mpsc_cost_horizon, decay_factor)

        self.output_dir = output_dir
        self.uncertified_controller = None
        self.skip_checks = False

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

        v_L = opti.parameter(self.model.nu, self.mpsc_cost_horizon)

        opti_dict['v_L'] = v_L

        cost = (u_L - next_u).T @ (u_L - next_u)
        for h in range(1, self.mpsc_cost_horizon):
            cost += (self.decay_factor**h) * (v_L[:, h] - v_var[:, h]).T @ (v_L[:, h] - v_var[:, h])

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

        if isinstance(self.uncertified_controller.env, VecEnv):
            uncert_env = self.uncertified_controller.env.envs[0]
        else:
            uncert_env = self.uncertified_controller.env

        v_L = np.zeros((self.model.nu, self.mpsc_cost_horizon))

        if isinstance(self.uncertified_controller, PID):
            self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_curr.npy')
            self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_prev.npy')

        for h in range(self.mpsc_cost_horizon):
            next_step = min(iteration + h, self.env.X_GOAL.shape[0] - 1)
            # Concatenate goal info (goal state(s)) for RL
            extended_obs = self.env.extend_obs(obs, next_step + 1)

            info = {
                'current_step': next_step,
                'constraint_values': np.concatenate([self.get_constraint_value(con, obs) for con in self.env.constraints.state_constraints])
            }

            action = self.uncertified_controller.select_action(obs=extended_obs, info=info)

            if uncert_env.NORMALIZED_RL_ACTION_SPACE:
                if self.env.NAME == Environment.CARTPOLE:
                    action = uncert_env.action_scale * action
                elif self.env.NAME == Environment.QUADROTOR:
                    action = (1 + uncert_env.norm_act_scale * action) * uncert_env.hover_thrust

            action = np.clip(action, self.env.physical_action_bounds[0], self.env.physical_action_bounds[1])

            if h == 0 and np.linalg.norm(uncertified_action - action) >= 0.001 and not self.skip_checks:
                raise ValueError(f'[ERROR] Mismatch between unsafe controller and MPSC guess. Uncert: {uncertified_action}, Guess: {action}, Diff: {np.linalg.norm(uncertified_action - action)}.')

            v_L[:, h:h + 1] = action.reshape((self.model.nu, 1))

            obs = np.squeeze(self.model.fd_func(x0=obs, p=action)['xf'].toarray())

        if isinstance(self.uncertified_controller, PID):
            self.uncertified_controller.load(f'{self.output_dir}/temp-data/saved_controller_curr.npy')
            self.uncertified_controller.save(f'{self.output_dir}/temp-data/saved_controller_prev.npy')

        return v_L

    def get_constraint_value(self, con, state):
        '''Gets the value of a constraint given the state.

        Args:
            con (Constraint): The constraint.
            state (ndarray): The state to be tested.

        Returns:
            value (float): The value of the constraint at the given state.
        '''
        return np.round_(np.atleast_1d(np.squeeze(con.sym_func(np.array(state, ndmin=1)))), decimals=con.decimals)
