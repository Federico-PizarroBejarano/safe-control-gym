'''A cost that penalizes rate of change of the MPC input sequence.'''

from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.abstract_cost import MPSC_COST


class REGULARIZED_COST(MPSC_COST):
    '''Regularized Cost.'''

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

        prev_u = opti.parameter(self.model.nu, 1)
        opti_dict['prev_u'] = prev_u

        gamma = 1

        cost = (u_L - next_u).T @ (u_L - next_u)
        for h in range(0, self.mpsc_cost_horizon):
            if h == 0:
                cost += gamma * (v_var[:, 0] - prev_u).T @ (v_var[:, 0] - prev_u)
            else:
                cost += gamma * (self.decay_factor**h) * (v_var[:, h] - v_var[:, h - 1]).T @ (v_var[:, h] - v_var[:, h - 1])
        return cost

    def prepare_cost_variables(self, opti_dict, obs, iteration):
        '''Prepares all the symbolic variable initial values for the next optimization.

        Args:
            opti_dict (dict): The dictionary of optimization variables.
            obs (ndarray): Current state/observation.
            iteration (int): The current iteration, used for trajectory tracking.
        '''

        opti = opti_dict['opti']
        prev_u = opti_dict['prev_u']

        opti.set_value(prev_u, opti_dict['prev_u_val'])
