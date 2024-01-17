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

import pickle

import casadi as cs
import cvxpy as cp
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template.acados_model import AcadosModel
from pytope import Polytope
from scipy.linalg import block_diag, solve_discrete_are, sqrtm

from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.envs.constraints import BoundedConstraint, ConstrainedVariableType
from safe_control_gym.safety_filters.cbf.cbf_utils import cartesian_product
from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class.'''

    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 integration_algo: str = 'rk4',
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 n_samples: int = 600,
                 cost_function: Cost_Function = Cost_Function.ONE_STEP_COST,
                 mpsc_cost_horizon: int = 5,
                 decay_factor: float = 0.85,
                 soften_constraints: bool = False,
                 slack_cost: float = 250,
                 **kwargs
                 ):
        '''Initialize the MPSC.

        Args:
            env_func (partial BenchmarkEnv): Environment for the task.
            horizon (int): The MPC horizon.
            integration_algo (str): The algorithm used for integrating the dynamics,
                either 'rk4', 'rk', or 'cvodes'.
            warmstart (bool): If the previous MPC soln should be used to warmstart the next mpc step.
            additional_constraints (list): List of additional constraints to consider.
            use_terminal_set (bool): Whether to use a terminal set constraint or not.
            n_samples (int): The number of state/action pairs to test when determining w_func.
            cost_function (Cost_Function): A string (from Cost_Function) representing the cost function to be used.
            mpsc_cost_horizon (int): How many steps forward to check for constraint violations.
            decay_factor (float): How much to discount future costs.
        '''

        self.model_bias = None

        self.n = 6
        self.m = 2
        self.q = 6

        super().__init__(env_func, horizon, q_lin, r_lin, integration_algo, warmstart, additional_constraints, use_terminal_set, cost_function, mpsc_cost_horizon, decay_factor, **kwargs)

        self.n_samples = n_samples
        self.soften_constraints = soften_constraints
        self.slack_cost = slack_cost

        state_lower_bounds = self.env.constraints.state_constraints[0].lower_bounds
        state_upper_bounds = self.env.constraints.state_constraints[0].upper_bounds
        self.state_constraint = BoundedConstraint(self.env, state_lower_bounds[[0,1,2,3,6,7]], state_upper_bounds[[0,1,2,3,6,7]], ConstrainedVariableType.STATE, active_dims=[0,1,2,3,6,7])
        self.input_constraint = BoundedConstraint(self.env, [-0.25, -0.25], [0.25, 0.25], ConstrainedVariableType.INPUT, active_dims=[0, 1])

        [self.X_mid, L_x, l_x] = self.box2polytopic(self.state_constraint)
        [self.U_mid, L_u, l_u] = self.box2polytopic(self.input_constraint)

        # number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = np.vstack((L_x, np.zeros((p_u, self.n))))
        self.L_u = np.vstack((np.zeros((p_x, self.m)), L_u))
        self.l_xu = np.concatenate([l_x, l_u])

        self.model.X_EQ = np.zeros((self.n,))
        self.model.U_EQ = np.zeros((self.m,))

    def set_dynamics(self):
        '''Compute the discrete dynamics.'''
        self.Ad = np.array([[ 1,      0.04,    0,     0,        0,       0.008],
                            [ 0,      1,       0,     0,        0,       0.365],
                            [ 0,      0,       1,     0.04,    -0.008,   0    ],
                            [ 0,      0,       0,     1,       -0.365,   0    ],
                            [ 0,      0,       0,     0.001,    0.815,  -0.003],
                            [ 0,     -0.001,   0,     0,       -0.003,   0.815]])

        self.Bd = np.array([[ 0,      0     ],
                            [ 0,      0.037 ],
                            [ 0,      0     ],
                            [-0.037,  0     ],
                            [ 0.205,  0     ],
                            [ 0,      0.205 ]])

        self.Ac = np.array([[0,  1,      0,  0,       0,         0.0125],
                            [0,  0.0052, 0,  0,       0.0179,   10.0887],
                            [0,  0,      0,  1,      -0.0125,    0     ],
                            [0,  0,      0,  0.0052,-10.0887,   -0.0179],
                            [0,  0,      0,  0.0276, -5.1084,   -0.0920],
                            [0, -0.0276, 0,  0,      -0.0920,   -5.1084]])

        self.Bc = np.array([[ 0,       -0.0129],
                            [-0.0025,  -0.1444],
                            [ 0.0129,   0     ],
                            [ 0.1444,   0.0025],
                            [ 5.6668,   0.0101],
                            [ 0.0101,   5.6668]])

        delta_x = cs.MX.sym('delta_x', self.n, 1)
        delta_u = cs.MX.sym('delta_u', self.m, 1)

        linear_sys = self.Ad @ delta_x + self.Bd @ delta_u
        if self.model_bias is not None:
            linear_sys = linear_sys + self.model_bias
        dynamics_func = cs.Function('fd',
                                    [delta_x, delta_u],
                                    [linear_sys],
                                    ['x0', 'p'],
                                    ['xf'])

        self.dynamics_func = dynamics_func

    def learn(self,
              env=None,
              **kwargs
              ):
        '''Compute values used by the MPC.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''

        if env is None:
            env = self.training_env

        self.tolerance = 1e-4

        self.x_r = self.model.X_EQ
        self.u_r = self.model.U_EQ

        x_sym = cs.MX.sym('delta_x', self.n, 1)
        u_sym = cs.MX.sym('delta_u', self.m, 1)
        w_sym = cs.MX.sym('delta_w', self.q, 1)

        self.get_error_function(env=env)
        self.Ed = np.diag(self.max_w_per_dim)
        self.Ec = self.Ed / self.dt

        self.f = cs.Function('disc_f', [x_sym, u_sym, w_sym], [self.Ac @ x_sym + self.Bc @ u_sym + self.Ec @ w_sym], ['x', 'u', 'w'], ['disc_f'])
        self.disc_f = cs.Function('disc_f', [x_sym, u_sym, w_sym], [self.Ad @ x_sym + self.Bd @ u_sym + self.Ed @ w_sym], ['x', 'u', 'w'], ['disc_f'])

        self.synthesize_lyapunov()
        self.get_terminal_ingredients()

        self.L_x_sym = cs.MX(self.L_x)
        self.L_u_sym = cs.MX(self.L_u)
        self.l_sym = cs.MX(self.l_xu)
        self.setup_optimizer()

    def get_error_function(self, env):
        '''Computes the maximum disturbance found in the training environment.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''
        # Create set of error residuals.
        w = np.load('./models/traj_data/errors.npy')
        print('MEAN ERROR PER DIM:', np.mean(w, axis=0))
        self.model_bias = np.mean(w, axis=0)
        self.set_dynamics()

        w = w - np.mean(w, axis=0)
        normed_w = np.linalg.norm(w, axis=1)
        self.max_w_per_dim = np.minimum(np.max(w, axis=0), np.mean(w, axis=0) + 3 * np.std(w, axis=0))
        self.max_w = min(np.max(normed_w), np.mean(normed_w) + 3 * np.std(normed_w))

        print('MAX ERROR:', np.max(normed_w))
        print('STD ERROR:', np.mean(normed_w) + 3 * np.std(normed_w))
        print('MEAN ERROR:', np.mean(normed_w))
        print('MAX ERROR PER DIM:', np.max(w, axis=0))
        print('STD ERROR PER DIM:', np.mean(w, axis=0) + 3 * np.std(w, axis=0))
        print('TOTAL ERRORS BY CHANNEL:', np.sum(np.abs(w), axis=0))

    def synthesize_lyapunov(self):
        '''Synthesize the appropriate constants related to the lyapunov function of the system.'''
        # Incremental Lyapunov function: Find upper bound for S-procedure variable lambda
        lamb_lb = None
        lamb_ub = None

        lamb = 0.008  # lambda lower bound
        self.rho_c = 0.192  # tuning parameter determines how fast the lyapunov function contracts

        while lamb < 100:
            lamb = lamb * 2
            [X, Y, cost, constraints] = self.setup_tube_optimization(lamb)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                print(f'Attempting with lambda={lamb}.')
                cost = prob.solve(solver=cp.MOSEK, verbose=False)
                if prob.status == 'optimal' and cost != float('inf'):
                    print(f'Succeeded with cost={cost}.')
                    if lamb_lb is None:
                        lamb_lb = lamb
                    lamb_ub = lamb
                else:
                    raise Exception('Not optimal or cost is infinite.')
            except Exception as e:
                print('Error in optimization:', e)
                if lamb_lb is not None:
                    break

        # Incremental Lyapunov function: Determine optimal lambda
        lamb_lb = lamb_lb / 2
        lamb_ub = lamb_ub * 2

        num_candidates = 50

        lambda_candidates = np.logspace(np.log(lamb_lb) / np.log(10), np.log(lamb_ub) / np.log(10), num_candidates)
        cost_values = []

        for i in range(num_candidates):
            lambda_candidate = lambda_candidates[i]
            [X, Y, cost, constraints] = self.setup_tube_optimization(lambda_candidate)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                cost = prob.solve(solver=cp.MOSEK, verbose=False)
                if prob.status != 'optimal' or cost == float('inf'):
                    raise cp.SolverError
            except Exception as e:
                print('Error in optimization:', e)
                cost = float('inf')
            cost_values += [cost]

        best_index = cost_values.index(min(cost_values))
        best_lamb = lambda_candidates[best_index]
        [X, Y, cost, constraints] = self.setup_tube_optimization(best_lamb)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        cost = prob.solve(solver=cp.MOSEK, verbose=False)
        if prob.status != 'optimal' or cost == float('inf'):
            raise cp.SolverError

        # Resulting continuous-time parameters
        self.X = X.value
        self.P = np.linalg.pinv(self.X)
        self.K = Y.value @ self.P

        self.c_js = np.zeros(self.p)

        for j in range(self.p):
            self.c_js[j] = np.linalg.norm((self.L_x[j, :] + self.L_u[j, :] @ self.K) @ sqrtm(self.X))

        c_max = max(self.c_js)
        w_bar_c = np.sqrt(np.max(np.linalg.eig(self.Ec.T @ self.P @ self.Ec)[0]))

        # Get Discrete-time system values
        self.rho = np.exp(-self.rho_c * self.dt)
        self.w_bar = w_bar_c * (1 - self.rho) / self.rho_c  # even using rho_c from the paper yields different w_bar
        # self.w_bar = max(self.w_bar, self.max_w)
        horizon_multiplier = (1 - self.rho**self.horizon) / (1 - self.rho)
        self.s_bar_f = horizon_multiplier * self.w_bar
        # assert self.s_bar_f > self.max_w * horizon_multiplier, f'[ERROR] s_bar_f ({self.s_bar_f}) is too small with respect to max_w ({self.max_w}).'
        # assert self.max_w * horizon_multiplier < 1.0, '[ERROR] max_w is too large and will overwhelm terminal set.'
        self.gamma = 1 / c_max - self.s_bar_f

        self.delta_loc = (horizon_multiplier * self.w_bar)**2

        print(f'rho: {self.rho}')
        print(f'w_bar: {self.w_bar}')
        print(f's_bar_f: {self.s_bar_f}')
        print(f'gamma: {self.gamma}')

        self.check_decay_rate()
        self.check_lyapunov_func()

    def get_terminal_ingredients(self):
        '''Calculate the terminal ingredients of the MPC optimization. '''
        self.P_f = solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        btp = np.dot(self.Bd.T, self.P_f)
        self.K_f = -np.dot(np.linalg.inv(self.R + np.dot(btp, self.Bd)), np.dot(btp, self.Ad))
        # self.check_terminal_ingredients()
        # self.check_terminal_constraints()
        # self.get_terminal_constraint()

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

    def setup_tube_optimization(self, lamb):
        '''Sets up the optimization to find the lyapunov function.

        Args:
            lamb (float): The S-procedure constant.

        Returns:
            X (cp.Variable): The X variable in the optimization.
            Y (cp.Variable): The Y variable in the optimization.
            Cost (cp.Expression): The cost function expression.
            Constraints (list): The list of cvxpy expressions representing the constraints.
        '''

        X = cp.Variable((self.n, self.n), PSD=True, name='X', complex=False)
        Y = cp.Variable((self.m, self.n), name='Y', complex=False)

        Cost = -cp.log_det(X)

        Constraints = []

        AXBY = self.Ac @ X + self.Bc @ Y

        constraint_1 = AXBY + AXBY.T + 2 * self.rho_c * X
        constraint_2 = cp.bmat([[AXBY + AXBY.T + lamb * X, self.Ec], [self.Ec.T, -lamb * np.eye(self.q)]])

        Constraints += [constraint_1 << 0]
        Constraints += [constraint_2 << 0]

        for j in range(0, self.p):
            LXLY = self.L_x[j:j + 1, :] @ X + self.L_u[j:j + 1, :] @ Y

            constraint_3 = cp.bmat([[np.array([[1]]), LXLY], [LXLY.T, X]])
            Constraints += [constraint_3 >> 0]

        return X, Y, Cost, Constraints

    def randsphere(self, num, dim, r):
        '''This function returns an num by dim array in which
           each of the num rows has the dim Cartesian coordinates
           of a random point uniformly-distributed over the
           interior of an dim-dimensional hypersphere with
           radius r and center at the origin.

        Args:
            num (int): The number of vectors.
            dim (int): The dimension of the hypersphere.
            r (float): The radius of the hypersphere.

        Returns:
            vectors (ndarray): The resulting random points inside the hypersphere.
        '''

        vectors = []

        while len(vectors) < num:
            u = np.random.normal(0, 1, dim)  # an array of d normally distributed random variables
            norm = np.sum(u**2)**(0.5)
            radius = r * np.random.rand()**(1.0 / dim)
            vec = radius * u / norm
            vectors.append(vec)

        return np.vstack(vectors)

    def check_decay_rate(self):
        '''Check the decay rate.'''
        X_sqrt = sqrtm(self.X)
        P_sqrt = sqrtm(self.P)

        left_side = max(np.linalg.eig(X_sqrt @ (self.Ac + self.Bc @ self.K).T @ P_sqrt + P_sqrt @ (self.Ac + self.Bc @ self.K) @ X_sqrt)[0]) + 2 * self.rho_c
        assert left_side <= self.tolerance, f'[ERROR] The solution {left_side} is not within the tolerance {self.tolerance}'

    def check_lyapunov_func(self):
        '''Check the incremental Lyapunov function.'''

        # select the number of random vectors to check
        num_random_vectors = 10000

        # Sample random points inside the set V_delta(x, z) <= delta_loc
        delta_x = self.randsphere(num_random_vectors, self.n, self.delta_loc).T
        dx_transform = np.linalg.inv(sqrtm(self.P)) @ delta_x
        dx_transform = self.x_r[:, None] + dx_transform  # transform point from error to actual state

        # sample random disturbance bounded by max_w
        w_dist = self.randsphere(num_random_vectors, self.q, self.max_w).T

        # set arbitrary v that satisfies the constraints for testing
        # v = np.array(self.constraints.input_constraints[0].upper_bounds)/10
        v = 0

        # initialize counters
        num_valid = 0
        inside_set = 0
        is_invariant = 0

        for i in range(num_random_vectors):
            # get random state
            x_i = dx_transform[:, i]

            # set up control inputs (u_r is required to get f_kappa(0, 0) = 0)
            u_x = self.K @ x_i + v + self.u_r
            u_z = self.K @ self.x_r + v + self.u_r

            # get dynamics
            w_none = np.zeros((self.q, 1))
            x_dot = np.squeeze(self.f(x_i - self.X_mid, u_x - self.U_mid, w_none).toarray())
            z_dot = np.squeeze(self.f(self.x_r - self.X_mid, u_z - self.U_mid, w_none).toarray())

            # evaluate Lyapunov function and its time derivative
            V_d = (x_i - self.x_r).T @ self.P @ (x_i - self.x_r)
            dVdt = (x_i - self.x_r).T @ self.P @ (x_dot - z_dot)

            # Check incremental Lypaunov function condition
            if dVdt <= -self.rho_c * V_d:
                num_valid += 1

            # check if states are inside V_d(x_i, z) <= delta_loc
            if V_d <= self.delta_loc:
                inside_set += 1

            # get next state
            x_plus = np.squeeze(self.disc_f(x_i - self.X_mid, u_x - self.U_mid, w_dist[:, i]).toarray())
            V_d_plus = (x_plus - self.x_r).T @ self.P @ (x_plus - self.x_r)

            # check robust control invariance
            if V_d_plus <= self.delta_loc:
                is_invariant += 1

        print('NUM_VALID:', num_valid / num_random_vectors)
        print('INSIDE SET:', inside_set / num_random_vectors)
        print('IS INVARIANT:', is_invariant / num_random_vectors)

    def check_terminal_ingredients(self):
        '''Check the terminal ingredients.'''

        w_none = np.zeros((self.q, 1))
        num_random_vectors = 10000

        # Sample points from gamma^2 * unit sphere
        delta_x = self.randsphere(num_random_vectors, self.n, self.gamma**2).T

        # Transform sampled points into ellipsoid to span the candidate terminal
        # set and shift around reference point x_r
        dx_transform = np.linalg.inv(sqrtm(self.P_f)) @ delta_x
        dx_transform = self.x_r[:, None] + dx_transform

        # sample random disturbance bounded by max_w
        w_dist = self.randsphere(num_random_vectors, self.q, self.max_w).T

        # initialize counter
        num_valid = 0
        inside_set = 0

        for i in range(num_random_vectors):
            # get sampled vector
            x_i = dx_transform[:, i]

            # get terminal control input
            u = self.K_f @ (x_i - self.x_r) + self.u_r

            # simulate system using control input
            x_plus = np.squeeze(self.disc_f(x_i - self.X_mid, u - self.U_mid, w_none).toarray())

            # disturbed x_plus
            x_plus_noisy = np.squeeze(self.disc_f(x_i - self.X_mid, u - self.U_mid, w_dist[:, i]).toarray())

            # evaluate stage cost and terminal costs
            stage = (x_i - self.x_r).T @ self.Q @ (x_i - self.x_r)
            V_f = (x_i - self.x_r).T @ self.P_f @ (x_i - self.x_r)
            V_f_plus = (x_plus - self.x_r).T @ self.P_f @ (x_plus - self.x_r)

            # check Lyapunov condition for terminal cost
            if V_f_plus <= V_f - stage:
                num_valid += 1

            # check if noisy state is still in terminal set
            V_f_plus_noisy = (x_plus_noisy - self.x_r).T @ self.P_f @ (x_plus_noisy - self.x_r)
            if V_f_plus_noisy <= self.gamma**2:
                inside_set += 1

        print('NUM_VALID:', num_valid / num_random_vectors)
        print('INSIDE SET:', inside_set / num_random_vectors)

    def check_terminal_constraints(self,
                                   num_points: int = 40,
                                   ):
        '''
        Check if the provided terminal set is only contains valid states using a gridded approach.

        Args:
            num_points (int): The number of points in each dimension to check.

        Returns:
            valid_cbf (bool): Whether the provided CBF candidate is valid.
            infeasible_states (list): List of all states for which the QP is infeasible.
        '''

        # Determine if terminal set inside state constraints
        terminal_max = np.sqrt(np.diag(np.linalg.inv(self.P_f / self.gamma**2)))
        terminal_min = -np.sqrt(np.diag(np.linalg.inv(self.P_f / self.gamma**2)))

        max_bounds = np.zeros((self.n))
        min_bounds = np.zeros((self.n))
        for i in range(self.n):
            tighten_by_max = self.c_js[i * 2] * self.s_bar_f
            tighten_by_min = self.c_js[i * 2 + 1] * self.s_bar_f
            max_bounds[i] = 1.0 / self.L_x[i * 2, i] * (self.l_xu[i * 2] - tighten_by_max)
            min_bounds[i] = 1.0 / self.L_x[i * 2 + 1, i] * (self.l_xu[i * 2 + 1] - tighten_by_min)

        if np.any(terminal_max > max_bounds) or np.any(terminal_min < min_bounds):
            raise ValueError('Terminal set is not constrained within the constraint set.')

        # Determine if the maximum input is within input constraints
        x = cp.Variable((self.n, 1))
        C = np.linalg.cholesky(self.P_f).T
        cost = cp.Maximize(self.K_f[0, :] @ x)
        constraint = [cp.norm(C @ x) <= self.gamma]
        prob = cp.Problem(cost, constraint)
        max_input = prob.solve(solver=cp.MOSEK)

        max_bounds = np.zeros((self.m))
        min_bounds = np.zeros((self.m))
        for i in range(self.m):
            tighten_by_max = self.c_js[self.n * 2 + i * 2] * self.s_bar_f
            tighten_by_min = self.c_js[self.n * 2 + i * 2 + 1] * self.s_bar_f
            max_bounds[i] = 1.0 / self.L_u[self.n * 2 + i * 2, i] * (self.l_xu[self.n * 2 + i * 2] - tighten_by_max)
            min_bounds[i] = 1.0 / self.L_u[self.n * 2 + i * 2 + 1, i] * (self.l_xu[self.n * 2 + i * 2 + 1] - tighten_by_min)

        if np.any(max_input + self.u_r > max_bounds + self.U_mid) or np.any(-max_input + self.u_r < min_bounds + self.U_mid):
            raise ValueError(f'Terminal controller causes inputs (max_input: {-max_input+self.u_r[0]}/{max_input+self.u_r[0]}) outside of input constraints (constraints: {min_bounds[0] + self.U_mid[0]}/{max_bounds[0] + self.U_mid[0]}).')

        # Make sure that every vertex is checked
        num_points = max(2 * self.n, num_points + num_points % (2 * self.n))
        num_points_per_dim = num_points // self.n

        # Create the lists of states to check
        states_to_sample = [np.linspace(self.X_mid[i], terminal_max[i] + self.X_mid[i], num_points_per_dim) for i in range(self.n)]
        states_to_check = cartesian_product(*states_to_sample)

        num_states_inside_set = 0
        failed_checks = 0
        failed_29a = 0
        failed_29b = 0
        failed_29d = 0

        for state in states_to_check:
            terminal_cost = (state - self.X_mid).T @ self.P_f @ (state - self.X_mid)
            in_terminal_set = terminal_cost < self.gamma**2

            if in_terminal_set:
                num_states_inside_set += 1
                failed = False

                # Testing condition 29a
                stable_input = self.K_f @ (state - self.x_r) + self.u_r
                next_state = np.squeeze(self.disc_f(state - self.X_mid, stable_input - self.U_mid, np.zeros((self.q, 1))).toarray())
                stage_cost = (state.T - self.X_mid) @ self.Q @ (state - self.X_mid)
                next_terminal_cost = (next_state - self.X_mid).T @ self.P_f @ (next_state - self.X_mid)

                if terminal_cost - stage_cost != 0 and next_terminal_cost / (terminal_cost - stage_cost) > 1.01:
                    failed_29a += 1
                    failed = True

                # Testing condition 29b
                num_disturbances = 100
                disturbances = self.randsphere(num_disturbances, self.n, self.max_w).T
                for w in range(num_disturbances):
                    disturbed_state = next_state + disturbances[:, w]
                    terminal_cost = (disturbed_state - self.X_mid).T @ self.P_f @ (disturbed_state - self.X_mid)
                    in_terminal_set = terminal_cost < self.gamma**2

                    if not in_terminal_set:
                        failed_29b += 1
                        failed = True
                        break

                # Testing condition 29d
                for j in range(self.p):
                    constraint_satisfaction = self.L_x[j, :] @ (state - self.X_mid) + self.L_u[j, :] @ (stable_input - self.U_mid) - self.l_xu[j] + self.c_js[j] * self.s_bar_f <= 0
                    if not constraint_satisfaction:
                        failed_29d += 1
                        failed = True
                        break

                if failed:
                    failed_checks += 1

        print(f'Number of states checked: {len(states_to_check)}')
        print(f'Number of states inside terminal set: {num_states_inside_set}')
        print(f'Number of checks failed: {failed_checks}')
        print(f'Number of checks failed due to 29a: {failed_29a}')
        print(f'Number of checks failed due to 29b: {failed_29b}')
        print(f'Number of checks failed due to 29d: {failed_29d}')

    def get_terminal_constraint(self):
        '''Calculates the terminal set as a linear constraint'''
        positions = np.linspace(-2, 2, 100)
        velocities = np.linspace(-1, 1, 100)
        interior_points = []
        for x in positions:
            for v in velocities:
                state = np.array([x, v])
                num_disturbances = 100
                disturbances = self.randsphere(num_disturbances, self.n, self.max_w).T
                for w in range(num_disturbances):
                    disturbed_state = state + disturbances[:, w]
                    if disturbed_state @ self.P_f @ disturbed_state <= self.gamma**2:
                        interior_points.append(state)
        self.terminal_set = Polytope(interior_points)
        self.terminal_set.minimize_V_rep()

        self.terminal_A = self.terminal_set.A
        self.terminal_b = self.terminal_set.b

    def load(self,
             path,
             ):
        '''Load values used by the MPSC.

        Args:
            path (str): Path to the required file.
        '''

        with open(path, 'rb') as f:
            parameters = pickle.load(f)

        self.rho_c = parameters['rho_c']
        self.X = parameters['X']
        self.K = parameters['K']
        self.P = parameters['P']
        self.delta_loc = parameters['delta_loc']
        self.rho = parameters['rho']
        self.s_bar_f = parameters['s_bar_f']
        self.w_bar = parameters['w_bar']
        self.max_w = parameters['max_w']
        self.c_js = parameters['c_js']
        self.gamma = parameters['gamma']
        self.P_f = parameters['P_f']
        self.K_f = parameters['K_f']
        self.model_bias = parameters['model_bias']

        self.set_dynamics()
        # self.terminal_A = parameters['terminal_A']
        # self.terminal_b = parameters['terminal_b']

        self.L_x_sym = cs.MX(self.L_x)
        self.L_u_sym = cs.MX(self.L_u)
        self.l_sym = cs.MX(self.l_xu)

        self.setup_optimizer()

    def save(self, path):
        '''Save values used by the MPSC.

        Args:
            path (str): Name of the file to be created.
        '''

        parameters = {}
        parameters['rho_c'] = self.rho_c
        parameters['X'] = self.X
        parameters['K'] = self.K
        parameters['P'] = self.P
        parameters['delta_loc'] = self.delta_loc
        parameters['rho'] = self.rho
        parameters['s_bar_f'] = self.s_bar_f
        parameters['w_bar'] = self.w_bar
        parameters['max_w'] = self.max_w
        parameters['c_js'] = self.c_js
        parameters['gamma'] = self.gamma
        parameters['P_f'] = self.P_f
        parameters['K_f'] = self.K_f
        parameters['model_bias'] = self.model_bias

        # parameters['terminal_A'] = self.terminal_A
        # parameters['terminal_b'] = self.terminal_b

        with open(path, 'wb') as f:
            pickle.dump(parameters, f)

    def setup_casadi_optimizer(self):
        '''Setup the certifying MPC problem.'''

        # Horizon parameter.
        horizon = self.horizon
        nx, nu = self.n, self.m
        # Define optimizer and variables.
        opti = cs.Opti('conic')
        # States.
        z_var = opti.variable(nx, horizon + 1)
        # Inputs.
        v_var = opti.variable(nu, horizon)
        # Lyapunov bound.
        s_var = opti.variable(1, horizon + 1)
        # Certified input.
        next_u = opti.variable(nu, 1)
        # Desired input.
        u_L = opti.parameter(nu, 1)
        # Current observed state.
        x_init = opti.parameter(nx, 1)
        # Reference trajectory and predicted LQR gains
        if self.env.TASK == Task.STABILIZATION:
            X_GOAL = opti.parameter(1, nx)
        elif self.env.TASK == Task.TRAJ_TRACKING:
            X_GOAL = opti.parameter(self.horizon, nx)

        # Add slack variables
        slack = opti.variable(self.horizon, self.p)
        term_slack = opti.variable(1)
        slack_cost = 0

        for i in range(self.horizon):
            # Dynamics constraints
            next_state = self.dynamics_func(x0=z_var[:, i], p=v_var[:, i])['xf']
            opti.subject_to(z_var[:, i + 1] == next_state)

            # Lyapunov size increase
            opti.subject_to(s_var[:, i + 1] == self.rho * s_var[:, i] + self.max_w)
            opti.subject_to(s_var[:, i] <= self.s_bar_f)

            # Constraints
            soft_con_coeff = 10000
            for j in range(self.p):
                tighten_by = self.c_js[j] * s_var[:, i + 1]
                opti.subject_to(self.L_x_sym[j, :] @ (z_var[:, i + 1] - self.X_mid) + self.L_u_sym[j, :] @ (v_var[:, i] - self.U_mid) - self.l_sym[j] + tighten_by <= slack[i, j])
                slack_cost += soft_con_coeff * slack[i, j]**2
                opti.subject_to(slack[i, j] >= 0)

        # Final state constraints
        soft_term_coeff = 10000
        if self.use_terminal_set:
            opti.subject_to(cs.vec(self.terminal_A @ z_var[:, -1] - self.terminal_b) <= term_slack)
            slack_cost += soft_term_coeff * term_slack**2
            opti.subject_to(term_slack >= 0)

        # Initial state constraints
        opti.subject_to(z_var[:, 0] == x_init)
        opti.subject_to(s_var[:, 0] == 0)

        # Real input
        opti.subject_to(next_u == v_var[:, 0])

        # Create solver (IPOPT solver as of this version).
        opts = {'expand': False, 'printLevel': 'none'}
        opti.solver('qpoases', opts)
        self.opti_dict = {
            'opti': opti,
            'z_var': z_var,
            'v_var': v_var,
            's_var': s_var,
            'u_L': u_L,
            'x_init': x_init,
            'next_u': next_u,
            'X_GOAL': X_GOAL,
            'slack': slack,
            'slack_term': term_slack,
        }

        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = self.cost_function.get_cost(self.opti_dict)
        cost = cost + slack_cost
        opti.minimize(cost)
        self.opti_dict['cost'] = cost

    def setup_acados_optimizer(self):
        '''setup_optimizer_acados'''
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # Setup model
        model = AcadosModel()
        x = cs.MX.sym('x')
        y = cs.MX.sym('y')
        x_dot = cs.MX.sym('x_dot')
        y_dot = cs.MX.sym('y_dot')
        roll = cs.MX.sym('x_dot')
        pitch = cs.MX.sym('y_dot')
        x_sym = cs.vertcat(x, x_dot, y, y_dot, roll, pitch)

        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        u_sym = cs.vertcat(f1, f2)

        model.x = x_sym
        model.u = u_sym
        model.f_expl_expr = self.Ac @ x_sym + self.Bc @ u_sym

        x1_dot = cs.MX.sym('x1_dot')
        vx_dot = cs.MX.sym('vx_dot')
        y1_dot = cs.MX.sym('y1_dot')
        vy_dot = cs.MX.sym('vy_dot')
        roll_dot = cs.MX.sym('roll_dot')
        pitch_dot = cs.MX.sym('pitch_dot')
        xdot = cs.vertcat(x1_dot, vx_dot, y1_dot, vy_dot, roll_dot, pitch_dot)

        model.xdot = xdot
        model.f_impl_expr = model.xdot - model.f_expl_expr
        model.name = 'mpsf'
        ocp.model = model

        nx, nu = self.n, self.m
        ny = nx + nu

        ocp.dims.N = self.horizon

        # set cost module
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        Q_mat = np.zeros((nx, nx))
        ocp.cost.W_e = np.zeros((nx, nx))
        R_mat = np.eye(nu)
        ocp.cost.W = block_diag(Q_mat, R_mat)

        ocp.cost.Vx = np.zeros((ny, nx))
        ocp.cost.Vx[:nx, :] = np.eye(nx)
        ocp.cost.Vu = np.zeros((ny, nu))
        ocp.cost.Vu[nx:nx + nu, :] = np.eye(nu)
        ocp.cost.Vx_e = np.eye(nx)

        ocp.model.cost_y_expr = cs.vertcat(model.x, model.u)
        ocp.model.cost_y_expr_e = model.x

        # Updated on each iteration
        ocp.cost.yref = np.concatenate((self.model.X_EQ, self.model.U_EQ))
        ocp.cost.yref_e = self.model.X_EQ

        # set constraints
        ocp.constraints.constr_type = 'BGH'
        ocp.constraints.constr_type_e = 'BGH'

        ocp.constraints.x0 = self.model.X_EQ
        ocp.constraints.C = self.L_x
        ocp.constraints.D = self.L_u
        ocp.constraints.lg = -1000 * np.ones((self.p))
        ocp.constraints.ug = np.zeros((self.p))

        # Slack
        ocp.constraints.Jsg = np.eye(self.p)
        ocp.cost.Zu = np.array([1] * self.p)
        ocp.cost.Zl = np.array([1] * self.p)
        ocp.cost.zu = np.array([1] * self.p)
        ocp.cost.zl = np.array([1] * self.p)

        # Options
        ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'
        ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        ocp.solver_options.hpipm_mode = 'BALANCE'
        ocp.solver_options.integrator_type = 'ERK'
        ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # set prediction horizon
        ocp.solver_options.tf = self.dt * self.horizon

        solver_json = 'acados_ocp_mpsf.json'
        ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json, generate=True, build=True)

        for stage in range(self.mpsc_cost_horizon):
            ocp_solver.cost_set(stage, 'W', (self.cost_function.decay_factor**stage) * ocp.cost.W)

        for stage in range(self.mpsc_cost_horizon, self.horizon):
            ocp_solver.cost_set(stage, 'W', 0 * ocp.cost.W)

        s_var = np.zeros((self.horizon + 1))
        g = np.zeros((self.horizon, self.p))

        for i in range(self.horizon):
            s_var[i + 1] = self.rho * s_var[i] + self.max_w
            for j in range(self.p):
                tighten_by = self.c_js[j] * s_var[i + 1]
                g[i, j] = (self.l_xu[j] - tighten_by)
            g[i, :] += (self.L_x @ self.X_mid) + (self.L_u @ self.U_mid)
            ocp_solver.constraints_set(i, 'ug', g[i, :])

        self.ocp_solver = ocp_solver
