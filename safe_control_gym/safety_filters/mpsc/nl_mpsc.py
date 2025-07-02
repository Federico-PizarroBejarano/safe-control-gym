'''NL Model Predictive Safety Certification (NL MPSC).

The core idea is that any learning controller input can be either certificated as safe or, if not safe, corrected
using an MPC controller based on Robust NL MPC.

Based on
    * K.P. Wabsersich and M.N. Zeilinger 'Linear model predictive safety certification for learning-based control' 2019
      https://arxiv.org/pdf/1803.08552.pdf
    * J. Köhler, R. Soloperto, M. A. Müller, and F. Allgöwer, "A computationally efficient robust model predictive
      control framework for uncertain nonlinear systems -- extended version," IEEE Trans. Automat. Contr., vol. 66,
      no. 2, pp. 794 801, Feb. 2021, doi: 10.1109/TAC.2020.2982585. http://arxiv.org/abs/1910.12081
'''

import casadi as cs
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_model import AcadosModel
from scipy.linalg import block_diag

from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 num_drones: int = 1,
                 q_mpc: list = None,
                 r_mpc: list = None,
                 warmstart: bool = True,
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
            num_drones (int): The number of drones.
            q_mpc (list): The MPC cost function.
            r_mpc (list): The MPC cost function.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
            soften_constraints (bool): Whether to soften the constraints or not.
            slack_cost (float): The slack cost.
            max_w (float): The maximum model mismatch.
        '''

        self.model_bias = None
        self.num_drones = num_drones
        super().__init__(env_func, horizon, q_mpc, r_mpc, 'rk4', warmstart, None, False, cost_function, mpsc_cost_horizon, decay_factor, **kwargs)

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

        # Number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = np.vstack((L_x, np.zeros((p_u, self.n))))
        self.L_u = np.vstack((np.zeros((p_x, self.m)), L_u))
        self.l_xu = np.concatenate([l_x, l_u])

        self.setup_optimizer()

    def set_dynamics(self):
        '''Compute the discrete dynamics.'''
        def multi_drone_dynamics(x_all, u_all):
            '''
            Apply dynamics to multiple drones.

            Args:
                x_all: Stacked state vector [x1; x2; ...; xN] where xi is state of drone i
                u_all: Stacked input vector [u1; u2; ...; uN] where ui is input of drone i

            Returns:
                x_dot_all: Stacked derivative vector [x_dot1; x_dot2; ...; x_dotN]
            '''
            nx_single = self.model.nx
            nu_single = self.model.nu

            x_dots = []
            for i in range(self.num_drones):
                # Extract state and input for drone i
                x_i = x_all[i * nx_single:(i + 1) * nx_single]
                u_i = u_all[i * nu_single:(i + 1) * nu_single]

                # Apply single drone dynamics
                x_dot_i = self.model.fc_func(x_i, u_i)
                x_dots.append(x_dot_i)

            # Stack all derivatives
            return cs.vertcat(*x_dots)

        self.dynamics_func = multi_drone_dynamics

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
        raise NotImplementedError('Casadi not implemented')

    def setup_acados_optimizer(self):
        '''Setup the certifying MPC problem in acados.'''
        # Create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Setup model for multiple drones
        model = AcadosModel()

        # Create stacked state and input variables
        x_stack = []
        u_stack = []
        for i in range(self.num_drones):
            x_i = cs.MX.sym(f'x_{i}', self.model.nx)
            u_i = cs.MX.sym(f'u_{i}', self.model.nu)
            x_stack.append(x_i)
            u_stack.append(u_i)

        model.x = cs.vertcat(*x_stack)
        model.u = cs.vertcat(*u_stack)
        model.name = f'{self.env.NAME}_multi_drone'

        # Use the multi-drone dynamics
        model.f_expl_expr = self.dynamics_func(model.x, model.u)
        ocp.model = model

        nx, nu = self.model.nx * self.num_drones, self.model.nu * self.num_drones
        ny = nx + nu

        # Set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Create block diagonal cost matrices for multiple drones
        if self.mpc_mode:
            Q_multi = block_diag(*[self.Q for _ in range(self.num_drones)])
            R_multi = block_diag(*[self.R for _ in range(self.num_drones)])
            ocp.cost.W = block_diag(Q_multi, R_multi)
        else:
            Q_multi = np.zeros((nx, nx))
            R_multi = np.eye(nu)
            ocp.cost.W = block_diag(Q_multi, R_multi)

        ocp.cost.W_e = Q_multi
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:nx + nu, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Stack equilibrium points for all drones
        X_EQ_multi = np.tile(self.model.X_EQ, self.num_drones)
        U_EQ_multi = np.tile(self.model.U_EQ, self.num_drones)

        # Updated on each iteration
        ocp.cost.yref = np.concatenate((X_EQ_multi, U_EQ_multi))
        ocp.cost.yref_e = X_EQ_multi

        # Setup constraints - apply to each drone
        ocp.constraints.constr_type = 'BGH'

        ocp.constraints.x0 = X_EQ_multi

        # Create block diagonal constraint matrices for multiple drones
        L_x_multi = block_diag(*[self.L_x for _ in range(self.num_drones)])
        L_u_multi = block_diag(*[self.L_u for _ in range(self.num_drones)])

        ocp.constraints.C = L_x_multi
        ocp.constraints.D = L_u_multi

        # Stack constraint bounds for all drones
        p_multi = self.p * self.num_drones
        ocp.constraints.lg = -1000 * np.ones(p_multi)
        ocp.constraints.ug = np.zeros(p_multi)

        # Slack
        if self.soften_constraints:
            ocp.constraints.Jsg = np.eye(p_multi)
            slack_weights = np.tile([self.slack_cost] * self.model.nx * 2 + [self.slack_cost * 100] * self.model.nu * 2, self.num_drones)
            ocp.cost.Zu = slack_weights
            ocp.cost.Zl = slack_weights
            ocp.cost.zu = slack_weights
            ocp.cost.zl = slack_weights

        # Options
        ocp.solver_options.N_horizon = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 200

        solver_json = 'acados_ocp_mpsf.json'
        ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json, generate=True, build=True)

        if not self.mpc_mode:
            for stage in range(self.mpsc_cost_horizon):
                ocp_solver.cost_set(stage, 'W', (self.cost_function.decay_factor**stage) * ocp.cost.W)

            for stage in range(self.mpsc_cost_horizon, self.horizon):
                ocp_solver.cost_set(stage, 'W', 0 * ocp.cost.W)

        g = np.zeros((self.horizon, p_multi))

        # Stack constraint vectors for all drones
        X_mid_multi = np.tile(self.X_mid, self.num_drones)
        U_mid_multi = np.tile(self.U_mid, self.num_drones)
        l_xu_multi = np.tile(self.l_xu, self.num_drones)

        for i in range(self.horizon):
            for j in range(p_multi):
                # Apply tightening only to state constraints (first n*2 constraints per drone)
                local_constraint_idx = j % self.p
                tighten_by = (self.max_w * i) if local_constraint_idx < self.n * 2 else 0
                g[i, j] = (l_xu_multi[j] - tighten_by)
            g[i, :] += (L_x_multi @ X_mid_multi) + (L_u_multi @ U_mid_multi)
            ocp_solver.constraints_set(i, 'ug', g[i, :])

        self.ocp_solver = ocp_solver
