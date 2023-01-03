'''LQR Cost Function for Smooth MPSC. '''

import numpy as np
import casadi as cs

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST
from safe_control_gym.envs.benchmark_env import Task


class LQR_COST(MPSC_COST):
    '''LQR MPSC Cost Function. '''

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
                v_L = -self.gain @ (z_var[:, h] - X_GOAL.T + X_EQ) + self.env.symbolic.U_EQ
            elif self.env.TASK == Task.TRAJ_TRACKING:
                v_L = -self.gain @ (z_var[:, h] - X_GOAL[h, :].T + X_EQ) + self.env.symbolic.U_EQ
            cost += (self.decay_factor**h)*(v_L - v_var[:, h]).T @ (v_L - v_var[:, h])

        return cost
