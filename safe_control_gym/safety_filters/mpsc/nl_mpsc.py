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
                 q_mpc: list = None,
                 r_mpc: list = None,
                 warmstart: bool = True,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 soften_constraints: bool = False,
                 slack_cost: float = 250,
                 max_w: float = 0.002,
                 min_collision_distance: float = 0.2,
                 initial_state: np.ndarray = None,
                 teleop_vec: list = None,
                 true_teleop_vec: list = None,
                 sf_type: str = 'none',
                 mpc_mode: bool = False,
                 mpc_to_sf_ratio: float = 0.1,
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
            min_collision_distance (float): The minimum distance between drones.
            initial_state (np.ndarray): The initial state.
            teleop_vec (list): The safety filter vector.
            true_teleop_vec (list): The true safety filter vector.
            sf_type (str): The safety filter type.
            mpc_mode (bool): Whether to use MPC mode or not.
            mpc_to_sf_ratio (float): The ratio of the MPC cost to the SF cost.
        '''

        self.model_bias = None

        self.initial_state = initial_state
        self.teleop_vec = teleop_vec
        self.num_drones = len(teleop_vec)
        self.true_teleop_vec = true_teleop_vec
        self.sf_type = sf_type
        self.mpc_mode = mpc_mode
        self.mpc_to_sf_ratio = mpc_to_sf_ratio

        super().__init__(env_func, horizon, q_mpc, r_mpc, warmstart, cost_function, mpsc_cost_horizon, decay_factor, **kwargs)

        self.soften_constraints = soften_constraints
        self.slack_cost = slack_cost
        self.max_w = max_w
        self.min_collision_distance = min_collision_distance

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nx

        self.state_constraint = self.constraints.state_constraints[0]
        self.input_constraint = self.constraints.input_constraints[0]

        self.X_mid, L_x, l_x = self.box2polytopic(self.state_constraint)
        self.U_mid, L_u, l_u = self.box2polytopic(self.input_constraint)

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
            x_dots = []
            for i in range(self.num_drones):
                # Extract state and input for drone i
                x_i = x_all[i * self.model.nx:(i + 1) * self.model.nx]
                u_i = u_all[i * self.model.nu:(i + 1) * self.model.nu]

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

    def select_action(self, obs, info=None):
        '''Determine the action to take at the current timestep.

        Args:
            obs (ndarray): The observation at this timestep.
            info (dict): The info at this timestep.

        Returns:
            action (ndarray): The action chosen by the controller.
        '''

        action, _ = self.certify_action(obs, np.tile(self.U_EQ, self.num_drones), info=info)
        return action

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
        model.name = f'{self.env.NAME}_multi_drone{"_mpc" if self.mpc_mode else ""}'

        # Use the multi-drone dynamics
        model.f_expl_expr = self.dynamics_func(model.x, model.u)
        ocp.model = model

        nx, nu = self.model.nx * self.num_drones, self.model.nu * self.num_drones
        ny = nx + nu

        # Set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # Create block diagonal cost matrices for multiple drones
        R_sf = []
        Q_mpc, R_mpc = [], []
        for teleop_drone in self.teleop_vec:
            if teleop_drone:
                if self.sf_type in ['safe_swarm_basic', 'safe_swarm_advanced']:
                    R_sf.append(1000 * np.eye(self.model.nu))
                else:
                    R_sf.append(np.eye(self.model.nu))
                Q_mpc.append(np.zeros((self.model.nx, self.model.nx)))
                R_mpc.append(np.zeros((self.model.nu, self.model.nu)))
            else:
                if self.sf_type in ['safe_teleop_basic', 'safe_teleop_advanced']:
                    Q_mpc.append(np.zeros((self.model.nx, self.model.nx)))
                    R_mpc.append(1000 * np.eye(self.model.nu))
                else:
                    Q_mpc.append(self.Q)
                    R_mpc.append(self.R)
                R_sf.append(np.zeros((self.model.nu, self.model.nu)))
        W_mpc = (self.mpc_to_sf_ratio ** 0.5) * block_diag(*Q_mpc, *R_mpc)
        W_sf = (self.mpc_to_sf_ratio ** (-0.5)) * block_diag(np.zeros((self.model.nx * self.num_drones, self.model.nx * self.num_drones)), *R_sf)
        ocp.cost.W = W_mpc + W_sf
        ocp.cost.W_e = (W_mpc + W_sf)[:nx, :nx]
        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:nx + nu, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        # Stack equilibrium points for all drones
        X_EQ_multi = self.initial_state.flatten()
        U_EQ_multi = np.tile(self.model.U_EQ, self.num_drones)

        # Updated on each iteration
        ocp.cost.yref = np.concatenate((X_EQ_multi, U_EQ_multi))
        ocp.cost.yref_e = X_EQ_multi

        ocp.constraints.x0 = X_EQ_multi

        # Add box constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.C = block_diag(*[self.L_x] * self.num_drones)
        ocp.constraints.D = block_diag(*[self.L_u] * self.num_drones)
        ocp.constraints.lg = -1000 * np.ones((self.p * self.num_drones))
        ocp.constraints.ug = np.zeros((self.p * self.num_drones))

        # Add collision constraints
        collision_constraints = self.create_collision_constraints(model.x)
        model.con_h_expr = cs.vertcat(collision_constraints)

        # Stack constraint bounds for all drones
        num_collision_constraints = collision_constraints.shape[0]
        ocp.constraints.lh = self.min_collision_distance**2 * np.ones(num_collision_constraints)
        ocp.constraints.uh = 1000 * np.ones(num_collision_constraints)

        # Slack
        if self.soften_constraints:
            ocp.constraints.Jsg = np.eye(self.p * self.num_drones)
            ocp.constraints.Jsh = np.eye(num_collision_constraints)
            slack_multiplier = np.array(([1] * self.model.nx + [100] * self.model.nu) * self.num_drones * 2 + [1] * num_collision_constraints, dtype=float)
            if self.sf_type in ['safe_teleop_basic', 'safe_teleop_advanced']:
                for i, teleop_drone in enumerate(self.teleop_vec):
                    if not teleop_drone:
                        slack_multiplier[self.p * i:self.p * (i + 1)] /= 1000.0
            if self.sf_type in ['safe_swarm_basic', 'safe_swarm_advanced']:
                for i, teleop_drone in enumerate(self.teleop_vec):
                    if teleop_drone:
                        slack_multiplier[self.p * i:self.p * (i + 1)] /= 1000.0
            slack_weights = self.slack_cost * slack_multiplier
            ocp.cost.Zu = slack_weights
            ocp.cost.Zl = slack_weights
            ocp.cost.zu = slack_weights
            ocp.cost.zl = slack_weights
            ocp.cost.Zu_0 = slack_weights[:self.p * self.num_drones]
            ocp.cost.Zl_0 = slack_weights[:self.p * self.num_drones]
            ocp.cost.zu_0 = slack_weights[:self.p * self.num_drones]
            ocp.cost.zl_0 = slack_weights[:self.p * self.num_drones]

        # Options
        ocp.solver_options.N_horizon = self.horizon
        ocp.solver_options.tf = self.dt * self.horizon
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        ocp.solver_options.nlp_solver_max_iter = 200

        solver_json = f'acados_ocp_mpsf{"_mpc" if self.mpc_mode else ""}.json'
        ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json, generate=True, build=True)

        for stage in range(self.mpsc_cost_horizon):
            ocp_solver.cost_set(stage, 'W', (self.cost_function.decay_factor**stage) * W_sf + W_mpc)

        for stage in range(self.mpsc_cost_horizon, self.horizon):
            ocp_solver.cost_set(stage, 'W', W_mpc)

        g = np.zeros((self.horizon, self.p * self.num_drones))

        for i in range(self.horizon):
            for j in range(self.p * self.num_drones):
                tighten_by = (self.max_w * i) if (j % self.p < self.n * 2) else 0
                g[i, j] = (self.l_xu[j % self.p] - tighten_by)
            g[i, :] += np.tile((self.L_x @ self.X_mid) + (self.L_u @ self.U_mid), self.num_drones)
            ocp_solver.constraints_set(i, 'ug', g[i, :])

        self.ocp_solver = ocp_solver

    def create_collision_constraints(self, x_stack):
        '''Create collision avoidance constraints between drones.

        Args:
            x_stack (cs.MX): Stacked state vector for all drones

        Returns:
            collision_constraints (cs.MX): CasADi expression for collision constraints
        '''
        constraints = []
        pos_indices = np.array([0, 2, 4])  # x, y, z positions

        # Create collision avoidance constraints between all pairs of drones
        for i in range(self.num_drones):
            for j in range(i + 1, self.num_drones):
                if self.sf_type in ['safe_teleop_basic', 'safe_teleop_advanced'] and not (self.teleop_vec[i] or self.teleop_vec[j]):
                    continue
                if self.sf_type in ['safe_swarm_basic', 'safe_swarm_advanced'] and (self.teleop_vec[i] and self.teleop_vec[j]):
                    continue
                # Extract positions for drones i and j
                pos_i = x_stack[i * self.model.nx + pos_indices]
                pos_j = x_stack[j * self.model.nx + pos_indices]

                distance = cs.sumsqr(pos_i - pos_j)
                constraints.append(distance)

        return cs.vertcat(*constraints)
