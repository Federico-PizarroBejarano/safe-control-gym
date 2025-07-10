'''Abstract Model Predictive Safety Certification (MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Tube MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
'''

from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain
from safe_control_gym.controllers.mpc.mpc_utils import get_cost_weight_matrix, reset_constraints
from safe_control_gym.safety_filters.base_safety_filter import BaseSafetyFilter
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.one_step_cost import ONE_STEP_COST
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.precomputed_cost import PRECOMPUTED_COST
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function, get_trajectory_on_horizon


class MPSC(BaseSafetyFilter, ABC):
    '''Abstract Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_mpc: list = None,
                 r_mpc: list = None,
                 warmstart: bool = True,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            q_mpc, r_mpc (list): Q and R gain matrices for linear controller.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
        '''
        self.use_acados = True
        # Store all params/args.
        for k, v in locals().items():
            if k != 'self' and k != 'kwargs' and '__' not in k:
                self.__dict__[k] = v

        super().__init__(env_func, **kwargs)

        np.random.seed(self.seed)

        # Setup the Environments.
        self.env = env_func(normalized_rl_action_space=False,
                            cost='quadratic')
        self.training_env = env_func(randomized_init=True,
                                     init_state=None,
                                     cost='quadratic',
                                     normalized_rl_action_space=False,
                                     )

        # Setup attributes.
        self.reset()
        self.dt = self.model.dt
        self.Q = get_cost_weight_matrix(q_mpc, self.model.nx)
        self.R = get_cost_weight_matrix(r_mpc, self.model.nu)

        self.X_EQ = np.zeros(self.model.nx)
        self.U_EQ = self.model.U_EQ

        self.set_dynamics()
        self.lqr_gain = -compute_lqr_gain(self.model, self.X_EQ, self.U_EQ, self.Q, self.R, discrete_dynamics=True)

        self.terminal_set = None
        self.prev_action = self.U_EQ

        self.constraints, self.state_constraints_sym, self.input_constraints_sym = reset_constraints(self.env.constraints.constraints)

        if cost_function == Cost_Function.ONE_STEP_COST:
            self.cost_function = ONE_STEP_COST()
            self.mpsc_cost_horizon = 1
            self.cost_function.mpsc_cost_horizon = 1
        elif cost_function == Cost_Function.PRECOMPUTED_COST:
            self.cost_function = PRECOMPUTED_COST(self.env, mpsc_cost_horizon, decay_factor, self.output_dir)
        else:
            raise NotImplementedError(f'The MPSC cost function {cost_function} has not been implemented')

    @abstractmethod
    def set_dynamics(self):
        '''Compute the dynamics.'''
        raise NotImplementedError

    def setup_optimizer(self):
        '''Setup the certifying MPC problem.'''
        if self.use_acados:
            self.setup_acados_optimizer()
        else:
            self.setup_casadi_optimizer()

    @abstractmethod
    def setup_casadi_optimizer(self):
        '''Setup the certifying MPC problem using CasADi.'''
        raise NotImplementedError

    @abstractmethod
    def setup_acados_optimizer(self):
        '''Setup the certifying MPC problem using ACADOS.'''
        raise NotImplementedError

    def before_optimization(self, obs):
        '''Setup the optimization.

        Args:
            obs (ndarray): Current state/observation.
        '''
        return

    def solve_optimization(self,
                           obs,
                           uncertified_action,
                           iteration=None,
                           ):
        '''Solve the MPC optimization problem for a given observation and uncertified input.

        Args:
            obs (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            action (ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''

        if self.use_acados:
            action, feasible = self.solve_acados_optimization(obs, uncertified_action, iteration)
        else:
            action, feasible = self.solve_casadi_optimization(obs, uncertified_action, iteration)
        return action, feasible

    def solve_casadi_optimization(self,
                                  obs,
                                  uncertified_action,
                                  iteration=None,
                                  ):
        '''Solve the MPC optimization problem for a given observation and uncertified input.

        Args:
            obs (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            action (ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''
        opti_dict = self.opti_dict
        opti = opti_dict['opti']
        z_var = opti_dict['z_var']
        v_var = opti_dict['v_var']
        next_u = opti_dict['next_u']
        u_L = opti_dict['u_L']
        x_init = opti_dict['x_init']
        X_GOAL = opti_dict['X_GOAL']
        opti.set_value(x_init, obs - self.X_EQ)
        opti.set_value(u_L, uncertified_action)
        clipped_X_GOAL = get_trajectory_on_horizon(self.env, iteration, self.horizon)
        opti.set_value(X_GOAL, clipped_X_GOAL)

        self.cost_function.prepare_cost_variables(opti_dict, obs, iteration)

        # Initial guess for optimization problem.
        if (self.warmstart and self.z_prev is not None and self.v_prev is not None):
            # Shift previous solutions by 1 step.
            z_guess = deepcopy(self.z_prev)
            v_guess = deepcopy(self.v_prev)
            z_guess[:, :-1] = z_guess[:, 1:]
            v_guess[:-1] = v_guess[1:]
            opti.set_initial(z_var, z_guess)
            opti.set_initial(v_var, v_guess)

        # Solve the optimization problem.
        try:
            sol = opti.solve()
            self.cost_prev = sol.value(opti_dict['cost'])
            x_val, u_val, next_u_val = sol.value(z_var), sol.value(v_var), sol.value(next_u)
            self.z_prev = x_val
            self.v_prev = u_val.reshape((self.model.nu), self.horizon)
            self.next_u_prev = next_u_val
            # Take the first one from solved action sequence.
            action = next_u_val
            self.prev_action = next_u_val
            feasible = True
        except Exception as e:
            print('Error Return Status:', opti.debug.return_status())
            print(e)
            feasible = False
            action = None
        return action, feasible

    def solve_acados_optimization(self,
                                  obs,
                                  uncertified_action,
                                  iteration=None,
                                  ):
        '''Solve the MPC optimization problem for a given observation and uncertified input.

        Args:
            obs (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            iteration (int): The current iteration, used for trajectory tracking.

        Returns:
            action (ndarray): The certified action.
            feasible (bool): Whether the safety filtering was feasible or not.
        '''

        if self.mpc_mode:
            clipped_X_GOAL = get_trajectory_on_horizon(self.env, iteration, self.horizon + 1)
            for stage in range(self.horizon):
                self.ocp_solver.cost_set(stage, 'yref', np.concatenate((clipped_X_GOAL[stage, :], np.tile(self.model.U_EQ, self.num_drones))))
            y_ref_e = clipped_X_GOAL[-1, :]
            self.ocp_solver.set(self.horizon, 'yref', y_ref_e)
        else:
            ocp_solver = self.ocp_solver
            ocp_solver.cost_set(0, 'yref', np.concatenate((np.zeros((self.model.nx * self.num_drones)), np.array(uncertified_action).reshape((self.model.nu * self.num_drones)))))

            if isinstance(self.cost_function, PRECOMPUTED_COST):
                uncert_input_traj = self.cost_function.calculate_unsafe_path(obs, uncertified_action, iteration)

                for stage in range(1, self.mpsc_cost_horizon):
                    ocp_solver.cost_set(stage, 'yref', np.concatenate((np.zeros((self.model.nx * self.num_drones)), uncert_input_traj[stage, :])))

        # Solve the optimization problem.
        action = self.ocp_solver.solve_for_x0(x0_bar=obs)
        self.cost_prev = self.ocp_solver.get_cost()
        x_val = np.zeros((self.horizon + 1, self.model.nx * self.num_drones))
        u_val = np.zeros((self.horizon, self.model.nu * self.num_drones))
        for i in range(self.horizon):
            x_val[i, :] = self.ocp_solver.get(i, 'x')
            u_val[i, :] = self.ocp_solver.get(i, 'u')
        x_val[self.horizon, :] = self.ocp_solver.get(self.horizon, 'x')
        self.z_prev = x_val.T
        self.v_prev = u_val.T
        # Take the first one from solved action sequence.
        self.prev_action = action
        feasible = True
        return action, feasible

    def certify_action(self,
                       current_state,
                       uncertified_action,
                       info=None,
                       ):
        '''Algorithm 1 from Wabsersich 2019.

        Args:
            current_state (ndarray): Current state/observation.
            uncertified_action (ndarray): The uncertified_controller's action.
            info (dict): The info at this timestep.

        Returns:
            certified_action (ndarray): The certified action
            success (bool): Whether the safety filtering was successful or not.
        '''
        uncertified_action = np.clip(uncertified_action, np.tile(self.env.physical_action_bounds[0], self.num_drones), np.tile(self.env.physical_action_bounds[1], self.num_drones))
        self.before_optimization(current_state)

        iteration = self.extract_step(info)
        certified_action, feasible = self.solve_optimization(current_state, uncertified_action, iteration)
        certified_action = np.squeeze(np.array(certified_action))

        self.results_dict['uncertified_action'].append(uncertified_action)
        self.results_dict['feasible'].append(feasible)
        self.results_dict['certified_action'].append(certified_action)
        self.results_dict['correction'].append(np.linalg.norm(certified_action - uncertified_action))

        return certified_action, feasible

    def setup_results_dict(self):
        '''Setup the results dictionary to store run information.'''
        self.results_dict = {}
        self.results_dict['feasible'] = []
        self.results_dict['kinf'] = []
        self.results_dict['uncertified_action'] = []
        self.results_dict['certified_action'] = []
        self.results_dict['correction'] = []

    def close(self):
        '''Cleans up resources.'''
        self.env.close()
        self.training_env.close()

    def reset(self):
        '''Prepares for training or evaluation.'''
        self.model = self.get_prior(self.env, self.prior_info)
        self.env.reset()
        self.training_env.reset()
        self.reset_before_run()

    def reset_before_run(self, env=None):
        '''Reinitialize just the controller before a new run.

        Args:
            env (BenchmarkEnv): The environment to be used for the new run.
        '''
        self.z_prev = None
        self.v_prev = None
        self.kinf = self.horizon - 1
        self.setup_results_dict()
