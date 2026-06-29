'''NL Model Predictive Safety Certification (NL MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Robust NL MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
    * J. Köhler, R. Soloperto, M. A. Müller, and F. Allgöwer, “A computationally efficient robust model predictive
      control framework for uncertain nonlinear systems -- extended version,” IEEE Trans. Automat. Contr., vol. 66,
      no. 2, pp. 794 801, Feb. 2021, doi: 10.1109/TAC.2020.2982585. http://arxiv.org/abs/1910.12081
'''

import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_model import AcadosModel
from scipy.linalg import block_diag

from safe_control_gym.controllers.mpc.mpc_utils import rk_discrete
from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 soften_constraints: bool = False,
                 slack_cost: float = 250,
                 max_w: float = 0.002,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            q_lin, r_lin (list): Q and R gain matrices for linear controller.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
            soften_constraints (bool): Whether to soften the constraints or not.
            slack_cost (float): The slack cost for the constraints.
            max_w (float): The maximum weight for the constraints.
        '''

        super().__init__(
            env_func=env_func,
            horizon=horizon,
            q_lin=q_lin,
            r_lin=r_lin,
            cost_function=cost_function,
            mpsc_cost_horizon=mpsc_cost_horizon,
            decay_factor=decay_factor,
            soften_constraints=soften_constraints,
            slack_cost=slack_cost,
            max_w=max_w,
            **kwargs)

        self.soften_constraints = soften_constraints
        self.slack_cost = slack_cost
        self.max_w = max_w

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nx

        self.state_constraint = self.constraints.state_constraints[0]
        self.input_constraint = self.constraints.input_constraints[0]

        [self.X_mid, L_x, l_x] = self.box2polytopic(self.state_constraint)
        [self.U_mid, L_u, l_u] = self.box2polytopic(self.input_constraint)

        # number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = np.vstack((L_x, np.zeros((p_u, self.n))))
        self.L_u = np.vstack((np.zeros((p_x, self.m)), L_u))
        self.l_xu = np.concatenate([l_x, l_u])

        self.setup_optimizer()

    def set_dynamics(self):
        '''Compute the discrete dynamics.'''
        self.dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)

    def box2polytopic(self, constraint):
        '''Convert constraints into an explicit polytopic form. This assumes that constraints contain the origin.

        Args:
            constraint (Constraint): The constraint to be converted.

        Returns:
            L (ndarray): The polytopic matrix.
            l (ndarray): Whether the constraint is active.
        '''

        Limit = []
        limit_active = []

        Z_mid = (constraint.upper_bounds + constraint.lower_bounds) / 2.0
        Z_limits = np.array([[constraint.upper_bounds[i] - Z_mid[i], constraint.lower_bounds[i] - Z_mid[i]] for i in range(constraint.upper_bounds.shape[0])])

        dim = Z_limits.shape[0]
        eye_dim = np.eye(dim)

        for constraint_id in range(0, dim):
            if Z_limits[constraint_id, 0] != -float('inf'):
                if Z_limits[constraint_id, 0] == 0:
                    limit_active += [0]
                    Limit += [-eye_dim[constraint_id, :]]
                else:
                    limit_active += [1]
                    factor = 1 / Z_limits[constraint_id, 0]
                    Limit += [factor * eye_dim[constraint_id, :]]

            if Z_limits[constraint_id, 1] != float('inf'):
                if Z_limits[constraint_id, 1] == 0:
                    limit_active += [0]
                    Limit += [eye_dim[constraint_id, :]]
                else:
                    limit_active += [1]
                    factor = 1 / Z_limits[constraint_id, 1]
                    Limit += [factor * eye_dim[constraint_id, :]]

        return Z_mid, np.array(Limit), np.array(limit_active)

    def setup_casadi_optimizer(self):
        '''Setup the certifying MPC problem in casadi.'''
        raise NotImplementedError('Removed Casadi optimizer for NL MPSC.')

    def setup_acados_optimizer(self, regenerate=True):
        '''Setup the certifying MPC problem in acados.

        Args:
            regenerate (bool): If False, reload the solver from existing generated code.
        '''
        if regenerate or not hasattr(self, '_acados_ocp'):
            ocp = AcadosOcp()

            # Setup model
            model = AcadosModel()
            model.x = self.model.x_sym
            model.u = self.model.u_sym
            model.name = self.env.NAME

            # Dynamics model
            model.f_expl_expr = self.model.fc_func(model.x, model.u)

            ocp.model = model

            nx, nu = self.model.nx, self.model.nu
            ny = nx + nu

            # Set cost module
            ocp.cost.cost_type = 'LINEAR_LS'
            Q_mat = np.zeros((nx, nx))
            R_mat = np.eye(nu)
            ocp.cost.W = block_diag(Q_mat, R_mat)
            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[nx:nx + nu, :] = np.eye(nu)

            # Updated on each iteration
            ocp.cost.yref = np.concatenate((self.model.X_EQ, self.model.U_EQ))

            # Setup constraints
            ocp.constraints.constr_type = 'BGH'
            ocp.constraints.x0 = self.model.X_EQ
            ocp.constraints.C = self.L_x
            ocp.constraints.D = self.L_u
            ocp.constraints.lg = -1000 * np.ones((self.p))
            ocp.constraints.ug = np.zeros((self.p))

            # Slack
            if self.soften_constraints:
                ocp.constraints.Jsg = np.eye(self.p)
                ocp.cost.Zu = np.array([self.slack_cost] * nx * 2 + [self.slack_cost * 100] * nu * 2)
                ocp.cost.Zl = np.array([self.slack_cost] * nx * 2 + [self.slack_cost * 100] * nu * 2)
                ocp.cost.zu = np.array([self.slack_cost] * nx * 2 + [self.slack_cost * 100] * nu * 2)
                ocp.cost.zl = np.array([self.slack_cost] * nx * 2 + [self.slack_cost * 100] * nu * 2)

            # Options
            ocp.solver_options.N_horizon = self.horizon
            ocp.solver_options.tf = self.dt * self.horizon
            ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
            ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
            ocp.solver_options.hpipm_mode = 'BALANCE'
            ocp.solver_options.integrator_type = 'ERK'
            ocp.solver_options.nlp_solver_type = 'SQP_RTI'
            ocp.solver_options.print_level = 0
            ocp.solver_options.qp_print_level = 0

            self._acados_ocp = ocp
        else:
            ocp = self._acados_ocp

        solver_json = getattr(self, 'acados_solver_json', 'acados_ocp_mpsf.json')
        self.ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json, generate=regenerate, build=regenerate)
        self._configure_acados_solver(self.ocp_solver, ocp)

    def reinstantiate_acados_solver(self):
        '''Reload the Acados solver from existing generated code without recompiling.'''
        self.setup_acados_optimizer(regenerate=False)

    def _configure_acados_solver(self, ocp_solver, ocp):
        '''Apply horizon-dependent costs and tightened constraints to an Acados solver.'''
        for stage in range(self.mpsc_cost_horizon):
            ocp_solver.cost_set(stage, 'W', (self.cost_function.decay_factor**stage) * ocp.cost.W)

        for stage in range(self.mpsc_cost_horizon, self.horizon):
            ocp_solver.cost_set(stage, 'W', 0 * ocp.cost.W)

        g = np.zeros((self.horizon, self.p))

        for i in range(self.horizon):
            for j in range(self.p):
                tighten_by = (self.max_w * i) if j < self.n * 2 else 0
                g[i, j] = (self.l_xu[j] - tighten_by)
            g[i, :] += (self.L_x @ self.X_mid) + (self.L_u @ self.U_mid)
            ocp_solver.constraints_set(i, 'ug', g[i, :])
