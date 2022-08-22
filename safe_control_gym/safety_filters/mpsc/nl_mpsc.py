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
import numpy as np
import casadi as cs
import cvxpy as cp
from scipy.linalg import sqrtm, solve_discrete_are

from safe_control_gym.safety_filters.mpsc.mpsc import MPSC
from safe_control_gym.controllers.mpc.mpc_utils import rk_discrete
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.envs.benchmark_env import Task, Environment


class NL_MPSC(MPSC):
    '''Model Predictive Safety Certification Class. '''


    def __init__(self,
                 env_func,
                 horizon: int = 10,
                 q_lin: list = None,
                 r_lin: list = None,
                 integration_algo: str = 'rk4',
                 warmstart: bool = True,
                 additional_constraints: list = None,
                 use_terminal_set: bool = True,
                 cost_function: str = Cost_Function.ONE_STEP_COST,
                 n_samples: int = 600,
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
            cost_function (str): A string (from Cost_Function) representing the cost function to be used.
            n_samples (int): The number of state/action pairs to test when determining w_func.
        '''

        super().__init__(env_func, horizon, q_lin, r_lin, integration_algo, warmstart, additional_constraints, use_terminal_set, cost_function)

        self.n_samples = n_samples

    def set_dynamics(self):
        '''Compute the discrete dynamics. '''

        if self.integration_algo == 'rk4':
            dynamics_func = rk_discrete(self.model.fc_func,
                                         self.model.nx,
                                         self.model.nu,
                                         self.dt)
        else:
            dynamics_func = cs.integrator('fd', self.integration_algo,
                                            {'x': self.model.x_sym,
                                            'p': self.model.u_sym,
                                            'ode': self.model.x_dot}, {'tf': self.dt}
                                    )

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

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nx
        self.tolerance = 1e-5

        if self.env.NAME == Environment.CARTPOLE:
            self.x_r = np.array([self.env.X_EQ[0], 0, 0, 0])
        elif self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 2:
            self.x_r = np.array([self.env.X_EQ[0], 0, self.env.X_EQ[2], 0, 0, 0])
        elif self.env.NAME == Environment.QUADROTOR and self.env.QUAD_TYPE == 3:
            self.x_r = np.array([self.env.X_EQ[0], 0, self.env.X_EQ[2], 0, self.env.X_EQ[4], 0, 0, 0, 0, 0, 0, 0])
        self.u_r = self.U_EQ

        x_sym = self.model.x_sym
        u_sym = self.model.u_sym
        w_sym = cs.MX.sym('delta_w', self.q, 1)

        self.get_error_function(env=env)
        self.E = np.diag(self.error_distrib)

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
        self.l = np.concatenate([l_x, l_u])

        self.f = cs.Function('f', [x_sym, u_sym, w_sym], [self.model.fc_func(x_sym+self.X_mid, u_sym+self.U_mid) + self.E @ w_sym], ['x', 'u', 'w'], ['f'])
        phi_1 = cs.Function('phi_1', [x_sym, u_sym, w_sym], [self.f(x_sym, u_sym, w_sym)], ['x', 'u', 'w'], ['phi_1'])
        phi_2 = cs.Function('phi_2', [x_sym, u_sym, w_sym], [self.f(x_sym + 0.5 * self.dt * phi_1(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_2'])
        phi_3 = cs.Function('phi_3', [x_sym, u_sym, w_sym], [self.f(x_sym + 0.5 * self.dt * phi_2(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_3'])
        phi_4 = cs.Function('phi_4', [x_sym, u_sym, w_sym], [self.f(x_sym + self.dt * phi_3(x_sym, u_sym, w_sym), u_sym, w_sym)], ['x', 'u', 'w'], ['phi_4'])
        rungeKutta = x_sym + self.dt / 6 * (phi_1(x_sym, u_sym, w_sym) + 2 * phi_2(x_sym, u_sym, w_sym) + 2 * phi_3(x_sym, u_sym, w_sym) + phi_4(x_sym, u_sym, w_sym))
        self.disc_f = cs.Function('disc_f', [x_sym, u_sym, w_sym], [rungeKutta+self.X_mid], ['x', 'u', 'w'], ['disc_f'])

        self.Ac = cs.Function('Ac', [x_sym, u_sym, w_sym], [cs.jacobian(self.f(x_sym, u_sym, w_sym), x_sym)], ['x', 'u', 'w'], ['Ac'])
        self.Bc = cs.Function('Bc', [x_sym, u_sym, w_sym], [cs.jacobian(self.f(x_sym, u_sym, w_sym), u_sym)], ['x', 'u', 'w'], ['Bc'])

        self.Ad = cs.Function('Ad', [x_sym, u_sym, w_sym], [cs.jacobian(self.disc_f(x_sym, u_sym, w_sym), x_sym)], ['x', 'u', 'w'], ['Ad'])
        self.Bd = cs.Function('Bd', [x_sym, u_sym, w_sym], [cs.jacobian(self.disc_f(x_sym, u_sym, w_sym), u_sym)], ['x', 'u', 'w'], ['Bd'])

        self.synthesize_lyapunov()
        self.get_terminal_ingredients()

        self.L_x = cs.MX(self.L_x)
        self.L_u = cs.MX(self.L_u)
        self.l = cs.MX(self.l)
        self.setup_optimizer()

    def get_error_function(self, env):
        '''Computes the maximum disturbance found in the training environment.

        Args:
            env (BenchmarkEnv): If a different environment is to be used for learning, can supply it here.
        '''

        if env is None:
            env = self.training_env

        # Create set of error residuals.
        w = np.zeros((self.n_samples, self.n))

        # Use uniform sampling of control inputs and states.
        for i in range(self.n_samples):
            init_state, _ = env.reset()
            if self.env.NAME == Environment.QUADROTOR:
                u = np.random.rand(self.model.nu)/20 - 1/40 + self.U_EQ
            else:
                u = env.action_space.sample() # Will yield a random action within action space.
            x_next_obs, _, _, _ = env.step(u)
            x_next_estimate = np.squeeze(self.dynamics_func(x0=init_state, p=u)['xf'].toarray())
            w[i, :] = x_next_obs - x_next_estimate

        normed_w = np.linalg.norm(w, axis=1)
        print('MAX ERROR:', np.max(normed_w))
        print('MEAN ERROR:', np.mean(normed_w))
        print('TOTAL ERRORS BY CHANNEL:', np.sum(np.abs(w), axis=0))
        self.error_distrib = np.sum(np.abs(w), axis=0) / np.linalg.norm(np.sum(np.abs(w), axis=0))
        self.max_w = np.max(normed_w)
        self.w_func = lambda x, u, s: self.max_w

    def synthesize_lyapunov(self):
        '''Synthesize the appropriate constants related to the lyapunov function of the system. '''
        ## Incremental Lyapunov function: Find upper bound for S-procedure variable lambda
        lamb_lb = None
        lamb_ub = None

        lamb = 0.008 # lambda lower bound
        rho_c = 0.192  # tuning parameter determines how fast the lyapunov function contracts

        Theta = [self.state_constraint.lower_bounds[-2], 0, self.state_constraint.upper_bounds[-2]]

        while lamb < 100:
            lamb = lamb * 2
            [X, Y, cost, constraints] = self.setup_tube_optimization(Theta, rho_c, lamb)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                print(f'Attempting with lambda={lamb}.')
                cost = prob.solve(solver=cp.MOSEK, verbose=True)
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

        ## Incremental Lyapunov function: Determine optimal lambda
        lamb_lb = lamb_lb/2
        lamb_ub = lamb_ub*2

        num_candidates = 50

        lambda_candidates = np.logspace(np.log(lamb_lb) / np.log(10), np.log(lamb_ub) / np.log(10), num_candidates)
        cost_values = []

        for i in range(num_candidates):
            lambda_candidate = lambda_candidates[i]
            [X, Y, cost, constraints] = self.setup_tube_optimization(Theta, rho_c, lambda_candidate)
            prob = cp.Problem(cp.Minimize(cost), constraints)
            try:
                cost = prob.solve(solver=cp.MOSEK, verbose=True)
                if prob.status != 'optimal' or cost == float('inf'):
                    raise cp.SolverError
            except Exception as e:
                print('Error in optimization:', e)
                cost = float('inf')
            cost_values += [cost]

        best_index = cost_values.index(min(cost_values))
        best_lamb = lambda_candidates[best_index]
        [X, Y, cost, constraints] = self.setup_tube_optimization(Theta, rho_c, best_lamb)
        prob = cp.Problem(cp.Minimize(cost), constraints)
        cost = prob.solve(solver=cp.MOSEK, verbose=True)
        if prob.status != 'optimal' or cost == float('inf'):
            raise cp.SolverError

        # Resulting continuous-time parameters
        X = X.value
        P = np.linalg.pinv(X)
        K = Y.value @ P

        self.c_js = np.zeros(self.p)

        for j in range(self.p):
            self.c_js[j] = np.linalg.norm((self.L_x[j, :] + self.L_u[j, :] @ K) @ sqrtm(X))

        c_max = max(self.c_js)
        w_bar_c = np.sqrt(np.max(np.linalg.eig(self.E.T @ P @ self.E)[0]))
        self.delta_loc = 1  # Tuning parameter determines how far from the origin you can be and still be nominal

        self.check_decay_rate(X, P, K, Theta, rho_c)
        self.check_lyapunov_func(P, K, rho_c)

        ## Get Discrete-time system values
        self.rho = np.exp(-rho_c * self.dt)
        self.w_bar = w_bar_c * (1 - self.rho) / rho_c  # even using rho_c from the paper yields different w_bar
        self.s_bar = (1 - self.rho**self.horizon) / (1 - self.rho) * self.w_bar
        self.gamma = 1 / c_max

    def get_terminal_ingredients(self):
        '''Calculate the terminal ingredients of the MPC optimization. '''
        ## Solve Lyapunov SDP using linearized discrete-time dynamics based on RK4 for terminal ingredients
        w_none = np.zeros((self.q, 1))
        A_lin = self.Ad(self.x_r - self.X_mid, self.u_r  - self.U_mid, w_none).toarray()
        B_lin = self.Bd(self.x_r - self.X_mid, self.u_r  - self.U_mid, w_none).toarray()

        self.P_f = solve_discrete_are(A_lin, B_lin, self.Q, self.R)
        btp = np.dot(B_lin.T, self.P_f)
        self.K_f = -np.dot(np.linalg.inv(self.R + np.dot(btp, B_lin)), np.dot(btp, A_lin))
        self.check_terminal_ingredients()

    def box2polytopic(self, constraint):
        '''Convert constraints into an explicit polytopic form. This assumes that constraints contain the origin.

        Args:
            constraint (Constraint): The constraint to be converted.

        Returns:
            L (ndarray): The polytopic matrix.
            l (ndarray): Whether the constraint is active.
        '''

        L = []
        l = []

        Z_mid = np.array([(constraint.upper_bounds[i] + constraint.lower_bounds[i])/2 for i in range(constraint.upper_bounds.shape[0])])
        Z_limits = np.array([[constraint.upper_bounds[i]-Z_mid[i], constraint.lower_bounds[i]-Z_mid[i]] for i in range(constraint.upper_bounds.shape[0])])

        dim = Z_limits.shape[0]
        eye_dim = np.eye(dim)

        for constraint_id in range(0, dim):
            if Z_limits[constraint_id, 0] != -float('inf'):
                if Z_limits[constraint_id, 0] == 0:
                    l += [0]
                    L += [-eye_dim[constraint_id, :]]
                else:
                    l += [1]
                    factor = 1 / Z_limits[constraint_id, 0]
                    L += [factor * eye_dim[constraint_id, :]]

            if Z_limits[constraint_id, 1] != float('inf'):
                if Z_limits[constraint_id, 1] == 0:
                    l += [0]
                    L += [eye_dim[constraint_id, :]]
                else:
                    l += [1]
                    factor = 1 / Z_limits[constraint_id, 1]
                    L += [factor * eye_dim[constraint_id, :]]

        return Z_mid, np.array(L), np.array(l)

    def setup_tube_optimization(self, Theta, rho_c, lamb):
        '''Sets up the optimization to find the lyapunov function.

        Args:
            Theta (list): The angles for which to test.
            rho_c (float): A tuning parameter determining how fast the lyapunov function contracts.
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

        x_test = np.zeros((self.n, 1))
        u_test = self.U_EQ
        w_test = np.zeros((self.q, 1))

        for i in range(3):
            x_test[-2] = Theta[i]
            A_theta = self.Ac(x_test, u_test - self.U_mid, w_test).toarray()
            B_theta = self.Bc(x_test, u_test - self.U_mid, w_test).toarray()

            AXBY = A_theta @ X + B_theta @ Y

            constraint_1 = AXBY + AXBY.T + 2 * rho_c * X
            constraint_2 = cp.bmat([[AXBY + AXBY.T + lamb * X, self.E], [self.E.T, -lamb * np.eye(self.q)]])

            Constraints += [constraint_1 << 0]
            Constraints += [constraint_2 << 0]

        for j in range(0, self.p):
            LXLY = self.L_x[j:j+1, :] @ X + self.L_u[j:j+1, :] @ Y

            constraint_3 = cp.bmat([[np.array([[1]]), LXLY], [LXLY.T, X]])
            Constraints += [constraint_3 >> 0]

        return X, Y, Cost, Constraints

    def randsphere(self, num, dim, r):
        '''This function returns an num by dim array, X, in which
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
            vec = np.random.uniform(size=(1, dim), low=-r, high=r)
            if np.linalg.norm(vec) < r:
                vectors.append(vec)

        return np.vstack(vectors)

    def check_decay_rate(self, X, P, K, Theta, rho_c):
        '''Check the decay rate.

        Args:
            X (ndarray): The X matrix.
            P (ndarray): The lyapunov matrix.
            K (ndarray): The gain matrix.
            Theta (list): The angles for which to test.
            rho_c (float): A tuning parameter determining how fast the lyapunov function contracts.
        '''

        x_test = np.zeros((self.n, 1))
        u_test = self.U_EQ
        w_test = np.zeros((self.q, 1))

        X_sqrt = sqrtm(X)
        P_sqrt = sqrtm(P)
        for angle in Theta:
            x_test[-2] = angle
            A_theta = self.Ac(x_test, u_test - self.U_mid, w_test).toarray()
            B_theta = self.Bc(x_test, u_test - self.U_mid, w_test).toarray()
            assert(max(np.linalg.eig(X_sqrt @ (A_theta + B_theta @ K).T @ P_sqrt + P_sqrt @ (A_theta + B_theta @ K) @ X_sqrt)[0]) + 2*rho_c <= self.tolerance)

    def check_lyapunov_func(self, P, K, rho_c):
        '''Check the incremental Lyapunov function.

        Args:
            P (ndarray): The lyapunov matrix.
            K (ndarray): The gain matrix.
            rho_c (float): A tuning parameter determining how fast the lyapunov function contracts.
        '''

        # select the number of random vectors to check
        num_random_vectors = 10000

        # Sample random points inside the set V_delta(x, z) <= delta_loc
        delta_x = self.randsphere(num_random_vectors, self.n, self.delta_loc).T
        dx_transform = np.linalg.inv(sqrtm(P)) @ delta_x
        dx_transform = self.x_r[:, None] + dx_transform  # transform point from error to actual state

        # sample random disturbance bounded by max_w
        w_dist = self.randsphere(num_random_vectors, self.q, self.max_w).T

        # set arbitrary v that satisfies the constraints for testing
        v = np.array(self.constraints.input_constraints[0].upper_bounds)/10

        # initialize counters
        num_valid = 0
        inside_set = 0
        is_invariant = 0

        for i in range(num_random_vectors):
            # get random state
            x_i = dx_transform[:, i]

            # set up control inputs (u_r is required to get f_kappa(0, 0) = 0)
            u_x = K @ x_i + v + self.u_r
            u_z = K @ self.x_r + v + self.u_r

            # get dynamics
            w_none = np.zeros((self.q, 1))
            x_dot = np.squeeze(self.f(x_i-self.X_mid, u_x-self.U_mid, w_none).toarray())
            z_dot = np.squeeze(self.f(self.x_r-self.X_mid, u_z-self.U_mid, w_none).toarray())

            # evaluate Lyapunov function and its time derivative
            V_d = (x_i - self.x_r).T @ P @ (x_i - self.x_r)
            dVdt = (x_i - self.x_r).T @ P @ (x_dot - z_dot)

            # Check incremental Lypaunov function condition
            if dVdt <= -rho_c * V_d:
                num_valid += 1

            # check if states are inside V_d(x_i, z) <= delta_loc
            if V_d <= self.delta_loc:
                inside_set += 1

            # get next state
            x_plus = np.squeeze(self.disc_f(x_i-self.X_mid, u_x-self.U_mid, w_dist[:, i]).toarray())
            V_d_plus = (x_plus - self.x_r).T @ P @ (x_plus - self.x_r)

            # check robust control invariance
            if V_d_plus <= self.delta_loc:
                is_invariant += 1

        print('NUM_VALID:', num_valid / num_random_vectors)
        print('INSIDE SET:', inside_set / num_random_vectors)
        print('IS INVARIANT:', is_invariant / num_random_vectors)

    def check_terminal_ingredients(self):
        '''Check the terminal ingredients. '''

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
            x_plus = np.squeeze(self.disc_f(x_i-self.X_mid, u-self.U_mid, w_none).toarray())

            # disturbed x_plus
            x_plus_noisy = np.squeeze(self.disc_f(x_i-self.X_mid, u-self.U_mid, w_dist[:, i]).toarray())

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

    def load(self,
             path,
             ):
        '''Load values used by the MPSC.

        Args:
            path (str): Path to the required file.
        '''

        self.n = self.model.nx
        self.m = self.model.nu
        self.q = self.model.nu

        state_constraint = self.constraints.state_constraints[0]
        input_constraint = self.constraints.input_constraints[0]

        [self.X_mid, L_x, l_x] = self.box2polytopic(state_constraint)
        [self.U_mid, L_u, l_u] = self.box2polytopic(input_constraint)

        # number of constraints
        p_x = l_x.shape[0]
        p_u = l_u.shape[0]
        self.p = p_x + p_u

        self.L_x = cs.MX(np.vstack((L_x, np.zeros((p_u, self.n)))))
        self.L_u = cs.MX(np.vstack((np.zeros((p_x, self.m)), L_u)))
        self.l = cs.MX(np.concatenate([l_x, l_u]))

        with open(path, 'rb') as f:
            parameters = pickle.load(f)

        self.rho = parameters['rho']
        self.s_bar = parameters['s_bar']
        self.w_bar = parameters['w_bar']
        self.max_w = parameters['max_w']
        self.w_func = lambda x, u, s: self.max_w
        self.c_js = parameters['c_js']
        self.gamma = parameters['gamma']
        self.P_f = parameters['P_f']

        self.setup_optimizer()

    def save(self, path):
        '''Save values used by the MPSC.

        Args:
            path (str): Name of the file to be created.
        '''

        parameters = {}
        parameters['rho'] = self.rho
        parameters['s_bar'] = self.s_bar
        parameters['w_bar'] = self.w_bar
        parameters['max_w'] = self.max_w
        parameters['c_js'] = self.c_js
        parameters['gamma'] = self.gamma
        parameters['P_f'] = self.P_f

        with open(path, 'wb') as f:
            pickle.dump(parameters, f)

    def setup_optimizer(self):
        '''Setup the certifying MPC problem. '''

        # Horizon parameter.
        horizon = self.horizon
        nx, nu = self.model.nx, self.model.nu
        # Define optimizer and variables.
        opti = cs.Opti()
        # States.
        z_var = opti.variable(nx, horizon + 1)
        # Inputs.
        v_var = opti.variable(nu, horizon)
        # Lyapunov bound.
        s_var = opti.variable(1, horizon + 1)
        # Error bound.
        w_var = opti.variable(1, horizon + 1)
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

        for i in range(self.horizon):
            # Dynamics constraints
            next_state = self.dynamics_func(x0=z_var[:,i], p=v_var[:,i])['xf']
            opti.subject_to(z_var[:, i+1] == next_state)

            # Lyapunov size increase
            s_var[:, i+1] = self.rho*s_var[:, i] + w_var[:, i]
            opti.subject_to(s_var[:, i] <= self.s_bar)
            opti.subject_to(w_var[:, i] <= self.w_bar)
            opti.subject_to(w_var[:, i] >= self.w_func(z_var[:,i], v_var[:,i], s_var[:,i]))

            # Constraints
            for j in range(self.p):
                tighten_by = self.c_js[j]*s_var[:, i+1]
                opti.subject_to(self.L_x[j, :] @ (z_var[:, i+1]-self.X_mid) + self.L_u[j, :] @ (v_var[:, i]-self.U_mid) - self.l[j] + tighten_by <= 0)

        # Final state constraints
        if self.use_terminal_set:
            if self.env.TASK == Task.STABILIZATION:
                terminal_cost = (z_var[:, -1] - X_GOAL.T).T @ self.P_f @ (z_var[:, -1] - X_GOAL.T)
            if self.env.TASK == Task.TRAJ_TRACKING:
                terminal_cost = (z_var[:, -1] - X_GOAL[-1, :].T).T @ self.P_f @ (z_var[:, -1] - X_GOAL[-1, :].T)
            opti.subject_to(terminal_cost <= self.gamma**2)

        # Initial state constraints
        opti.subject_to(z_var[:, 0] == x_init)
        opti.subject_to(s_var[:, 0] == 0)

        # Real input
        opti.subject_to(next_u == v_var[:, 0])

        # Create solver (IPOPT solver as of this version).
        opts = {'expand': False,
                'ipopt.print_level': 4,
                'ipopt.sb': 'yes',
                'ipopt.max_iter': 50,
                'print_time': 1}
        if self.integration_algo == 'rk4':
            opts['expand'] = True
        opti.solver('ipopt', opts)
        self.opti_dict = {
            'opti': opti,
            'z_var': z_var,
            'v_var': v_var,
            's_var': s_var,
            'w_var': w_var,
            'u_L': u_L,
            'x_init': x_init,
            'next_u': next_u,
            'X_GOAL': X_GOAL,
        }

        # Cost (# eqn 5.a, note: using 2norm or sqrt makes this infeasible).
        cost = self.cost_function.get_cost(self.opti_dict)
        opti.minimize(cost)
