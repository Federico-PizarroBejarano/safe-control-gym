'''LQR Cost Function for Smooth MPSC. '''

import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix
from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain
from safe_control_gym.envs.benchmark_env import Task


class LQR_COST(MPSC_COST):
    '''LQR MPSC Cost Function. '''

    def __init__(self,
                 env,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 ):
        '''Initialize the MPSC Cost.

        Args:
            env (BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
        '''

        self.env = env

        # Setup attributes.
        self.model = self.env.symbolic
        self.Q = get_cost_weight_matrix(q_lin, self.model.nx)
        self.R = get_cost_weight_matrix(r_lin, self.model.nu)

        if self.env.TASK == Task.STABILIZATION:
            self.gain = compute_lqr_gain(self.model, self.env.X_GOAL, self.env.U_GOAL,
                                         self.Q, self.R)

        self.horizon = horizon

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
        z_var = opti_dict['z_var']
        v_var = opti_dict['v_var']
        X_GOAL = opti_dict['X_GOAL']

        nu = self.model.nu
        nx = self.model.nx

        if 'X_EQ' in opti_dict:
            X_EQ = opti_dict['X_EQ']
        else:
            X_EQ = cs.MX(np.zeros(nx))

        # Predicted LQR gains
        if self.env.TASK == Task.STABILIZATION:
            gains = opti.parameter(nu, nx)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            gains = opti.parameter(self.horizon*nu, nx)

        opti_dict['gains'] = gains

        cost = (u_L - next_u).T @ (u_L - next_u)
        for h in range(1, self.horizon):
            if self.env.TASK == Task.STABILIZATION:
                v_L = -gains @ (z_var[:, h] - X_GOAL.T + X_EQ) + self.env.U_EQ
            elif self.env.TASK == Task.TRAJ_TRACKING:
                v_L = -gains[h*nu:h*nu+nu, :] @ (z_var[:, h] - X_GOAL[h, :].T + X_EQ) + self.env.U_EQ
            cost += (v_L - v_var[:, h]).T @ (v_L - v_var[:, h])

        return cost

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepares all the symbolic variable initial values for the next optimization.

        Args:
            opti_dict (dict): The dictionary of optimization variables.
            obs (ndarray): Current state/observation.
            iteration (int): The current iteration, used for trajectory tracking.
        '''

        opti = opti_dict['opti']
        gains = opti_dict['gains']

        if self.env.TASK == Task.TRAJ_TRACKING:
            expected_gains = self.calculate_gains(iteration)
            opti.set_value(gains, expected_gains)
        else:
            opti.set_value(gains, self.gain)

    def calculate_gains(self, iteration):
        '''Calculates the LQR gain at the current iteration when trajectory tracking.

        Args:
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            gains (ndarray): The gains for the whole horizon.
        '''

        nu = self.model.nu
        nx = self.model.nx

        gains = np.zeros((self.horizon*nu, nx))
        for h in range(self.horizon):
            next_iter = min(iteration+h, self.env.X_GOAL.shape[0]-1)
            gain = compute_lqr_gain(self.model, self.env.X_GOAL[next_iter],
                                        self.env.U_GOAL, self.Q, self.R)
            gains[h*nu:h*nu+nu, :] = gain

        return gains
