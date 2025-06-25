"""1D, 2D, and 3D quadrotor environment using PyBullet physics.

Based on UTIAS Dynamic Systems Lab's gym-pybullet-drones:
    * https://github.com/utiasDSL/gym-pybullet-drones
"""

import math
import os
from copy import deepcopy

import casadi as cs
import numpy as np
import pybullet as p
from gymnasium import spaces

from safe_control_gym.envs.benchmark_env import Cost, Task
from safe_control_gym.envs.constraints import GENERAL_CONSTRAINTS
from safe_control_gym.envs.disturbances import Downwash
from safe_control_gym.envs.gym_pybullet_drones.base_aviary import BaseAviary, Physics
from safe_control_gym.envs.gym_pybullet_drones.quadrotor_utils import (AttitudeControl, QuadType, cmd2pwm,
                                                                       pwm2rpm)
from safe_control_gym.envs.gym_pybullet_drones.trajectory_utils import _plot_trajectory
from safe_control_gym.math_and_models.symbolic_systems import SymbolicModel
from safe_control_gym.math_and_models.transformations import (csRotXYZ, get_quaternion_from_euler,
                                                              transform_trajectory)

script_dir = os.path.dirname(__file__)


class Quadrotor(BaseAviary):
    """1D, 2D, and 3D quadrotor environment task.

    Including symbolic model, constraints, randomization, adversarial disturbances,
    multiple cost functions, stabilization and trajectory tracking references.
    """

    NAME = 'quadrotor'
    AVAILABLE_CONSTRAINTS = deepcopy(GENERAL_CONSTRAINTS)

    DISTURBANCE_MODES = {  # Set at runtime by QUAD_TYPE
        'observation': {
            'dim': -1
        },
        'action': {
            'dim': -1
        },
        'dynamics': {
            'dim': -1
        },
        'downwash': {
            'dim': -1
        }
    }

    INERTIAL_PROP_RAND_INFO = {
        'M': {  # Nominal: 0.027
            'distrib': 'uniform',
            'low': 0.022,
            'high': 0.032
        },
        'Ixx': {  # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Iyy': {  # Nominal: 1.4e-5
            'distrib': 'uniform',
            'low': 1.3e-5,
            'high': 1.5e-5
        },
        'Izz': {  # Nominal: 2.17e-5
            'distrib': 'uniform',
            'low': 2.07e-5,
            'high': 2.27e-5
        },
        'beta_1': {  # Nominal: 18.11
            'distrib': 'uniform',
            'low': -4,
            'high': 4
        },
        'beta_2': {  # Nominal: 3.68
            'distrib': 'uniform',
            'low': -0.7,
            'high': 0.7
        },
        'alpha_1': {  # Nominal: -140.8
            'distrib': 'uniform',
            'low': -5,
            'high': 10
        },
        'alpha_2': {  # Nominal: -13.4
            'distrib': 'uniform',
            'low': -3,
            'high': 3
        },
        'alpha_3': {  # Nominal: 124.8
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_4': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_5': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_6': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_7': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_8': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        },
        'alpha_9': {  # Nominal: 0.0
            'distrib': 'uniform',
            'low': -5,
            'high': 5
        }
    }

    INIT_STATE_RAND_INFO = {
        'init_x': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_x_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_y': {
            'distrib': 'uniform',
            'low': -0.5,
            'high': 0.5
        },
        'init_y_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_z': {
            'distrib': 'uniform',
            'low': 0.1,
            'high': 1.5
        },
        'init_z_dot': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_phi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_theta': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_psi': {
            'distrib': 'uniform',
            'low': -0.3,
            'high': 0.3
        },
        'init_p': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_theta_dot': {  # TODO: replace with q.
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_q': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_r': {
            'distrib': 'uniform',
            'low': -0.01,
            'high': 0.01
        },
        'init_tau': {
            'distrib': 'uniform',
            'low': 0.033 * 9.81 - 0.01,
            'high': 0.033 * 9.81 + 0.01
        }
    }

    TASK_INFO = {
        'stabilization_goal': [0, 1],
        'stabilization_goal_tolerance': 0.05,
        'trajectory_type': 'circle',
        'num_cycles': 1,
        'trajectory_plane': 'zx',
        'trajectory_position_offset': [0.5, 0],
        'trajectory_scale': -0.5,
        'proj_point': [0, 0, 0.5],
        'proj_normal': [0, 1, 1],
    }

    def __init__(self,
                 init_state=None,
                 inertial_prop=None,
                 # custom args
                 quad_type: QuadType = QuadType.TWO_D,
                 norm_act_scale=0.1,
                 obs_goal_horizon=0,
                 rew_state_weight=1.0,
                 rew_act_weight=0.0001,
                 rew_exponential=True,
                 use_rl_reward_weights=False,
                 done_on_out_of_bound=True,
                 info_mse_metric_state_weight=None,
                 **kwargs
                 ):
        """Initialize a quadrotor environment.

        Args:
            init_state (ndarray, optional): The initial state of the environment, (z, z_dot) or (x, x_dot, z, z_dot theta, theta_dot).
            inertial_prop (ndarray, optional): The inertial properties of the environment (M, Ixx, Iyy, Izz).
            quad_type (QuadType, optional): The choice of motion type (1D along z, 2D in the x-z plane, or 3D).
            norm_act_scale (float): Scaling the [-1,1] action space around hover thrust when `normalized_action_space` is True.
            obs_goal_horizon (int): How many future goal states to append to observation.
            rew_state_weight (list/ndarray): Quadratic weights for state in rl reward.
            rew_act_weight (list/ndarray): Quadratic weights for action in rl reward.
            rew_exponential (bool): If to exponential negative quadratic cost to positive, bounded [0,1] reward.
            done_on_out_of_bound (bool): If to terminate when state is out of bound.
            info_mse_metric_state_weight (list/ndarray): Quadratic weights for state in mse calculation for info dict.
        """

        self.QUAD_TYPE = QuadType(quad_type)
        self.norm_act_scale = norm_act_scale
        self.obs_goal_horizon = obs_goal_horizon
        self.rew_state_weight = np.array(rew_state_weight, ndmin=1, dtype=float)
        self.Q = np.diag(self.rew_state_weight)
        self.rew_act_weight = np.array(rew_act_weight, ndmin=1, dtype=float)
        self.R = np.diag(self.rew_act_weight)
        self.rew_exponential = rew_exponential
        self.use_rl_reward_weights = use_rl_reward_weights
        self.done_on_out_of_bound = done_on_out_of_bound
        if info_mse_metric_state_weight is None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.info_mse_metric_state_weight = np.array([1, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 0, 0], ndmin=1, dtype=float)
            # elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            elif self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_BODY]:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
                self.info_mse_metric_state_weight = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), not implemented quad type.')
        else:
            if (self.QUAD_TYPE == QuadType.ONE_D and len(info_mse_metric_state_weight) == 2) or \
                    (self.QUAD_TYPE == QuadType.TWO_D and len(info_mse_metric_state_weight) == 6) or \
                    (self.QUAD_TYPE == QuadType.THREE_D and len(info_mse_metric_state_weight) == 12) or \
                    (self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE and len(info_mse_metric_state_weight) == 6) or \
                    (self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY and len(info_mse_metric_state_weight) == 6) or \
                    (self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S and len(info_mse_metric_state_weight) == 5) or \
                    (self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE and len(info_mse_metric_state_weight) == 12) or \
                    (self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10 and len(info_mse_metric_state_weight) == 10) or \
                    (self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY and len(info_mse_metric_state_weight) == 13):

                self.info_mse_metric_state_weight = np.array(info_mse_metric_state_weight, ndmin=1, dtype=float)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), wrong info_mse_metric_state_weight argument size.')

        # BaseAviary constructor, called after defining the custom args,
        # since some BenchmarkEnv init setup can be task(custom args)-dependent.
        super().__init__(init_state=init_state, inertial_prop=inertial_prop, **kwargs)

        # Store initial state info.
        self.INIT_STATE_LABELS = {
            QuadType.ONE_D: ['init_x', 'init_x_dot'],
            QuadType.TWO_D: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
            QuadType.TWO_D_ATTITUDE: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
            QuadType.TWO_D_ATTITUDE_BODY: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta', 'init_theta_dot'],
            QuadType.TWO_D_ATTITUDE_5S: ['init_x', 'init_x_dot', 'init_z', 'init_z_dot', 'init_theta'],
            QuadType.THREE_D: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                               'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_r'],
            QuadType.THREE_D_ATTITUDE: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                                        'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_r'],
            QuadType.THREE_D_ATTITUDE_10: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                                           'init_phi', 'init_theta', 'init_p', 'init_q'],
            QuadType.THREE_D_ATTITUDE_DELAY: ['init_x', 'init_x_dot', 'init_y', 'init_y_dot', 'init_z', 'init_z_dot',
                                              'init_phi', 'init_theta', 'init_psi', 'init_p', 'init_q', 'init_tau']
        }
        if init_state is None:
            for init_name in self.INIT_STATE_RAND_INFO:  # Default zero state.
                self.__dict__[init_name.upper()] = 0.
        else:
            if isinstance(init_state, np.ndarray):  # Full state as numpy array .
                for i, init_name in enumerate(self.INIT_STATE_LABELS[self.QUAD_TYPE]):
                    self.__dict__[init_name.upper()] = init_state[i]
            elif isinstance(init_state, dict):  # Partial state as dictionary.
                for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                    self.__dict__[init_name.upper()] = init_state.get(init_name, 0.)
            else:
                raise ValueError('[ERROR] in Quadrotor.__init__(), init_state incorrect format.')

        # Remove randomization info of initial state components inconsistent with quad type.
        for init_name in list(self.INIT_STATE_RAND_INFO.keys()):
            if init_name not in self.INIT_STATE_LABELS[self.QUAD_TYPE]:
                self.INIT_STATE_RAND_INFO.pop(init_name, None)
        # Remove randomization info of inertial components inconsistent with quad type.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # Do NOT randomize J for the 1D quadrotor.
            self.INERTIAL_PROP_RAND_INFO.pop('Ixx', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Iyy', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Izz', None)
        elif self.QUAD_TYPE == QuadType.TWO_D or \
                self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or \
                self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S or \
                self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
            # Only randomize Iyy for the 2D quadrotor.
            self.INERTIAL_PROP_RAND_INFO.pop('Ixx', None)
            self.INERTIAL_PROP_RAND_INFO.pop('Izz', None)

        # Override inertial properties of passed as arguments.
        if inertial_prop is None:
            pass
        elif self.QUAD_TYPE == QuadType.ONE_D and np.array(inertial_prop).shape == (1,):
            self.MASS = inertial_prop[0]
        elif self.QUAD_TYPE == QuadType.TWO_D and np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE and np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S and np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY and np.array(inertial_prop).shape == (2,):
            self.MASS, self.J[1, 1] = inertial_prop
        elif self.QUAD_TYPE == QuadType.THREE_D and np.array(inertial_prop).shape == (4,):
            self.MASS, self.J[0, 0], self.J[1, 1], self.J[2, 2] = inertial_prop
        elif isinstance(inertial_prop, dict):
            self.MASS = inertial_prop.get('M', self.MASS)
            self.J[0, 0] = inertial_prop.get('Ixx', self.J[0, 0])
            self.J[1, 1] = inertial_prop.get('Iyy', self.J[1, 1])
            self.J[2, 2] = inertial_prop.get('Izz', self.J[2, 2])
        else:
            raise ValueError('[ERROR] in Quadrotor.__init__(), inertial_prop incorrect format.')

        # Set goals for current task
        self.set_goals()

        # Set attitude controller if quadtype is QuadType.TWO_D_ATTITUDE
        # if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.THREE_D_ATTITUDE]:
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY,
                              QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10, QuadType.THREE_D_ATTITUDE_DELAY]:
            self.attitude_control = AttitudeControl(self.CTRL_TIMESTEP, self.PYB_TIMESTEP)

        # Set prior/symbolic info.
        if inertial_prop is None:
            self._setup_symbolic()
        else:
            self._setup_symbolic(inertial_prop)

        # initialize disturbance model
        if 'downwash' in self.disturbances:
            if self.DISTURBANCES['downwash'][0]['mode'] == 'fix':
                self.dw_model = Downwash(
                    init_pos=self.DISTURBANCES['downwash'][0]['pos'],
                    low=self.DISTURBANCES['downwash'][0]['low'],
                    high=self.DISTURBANCES['downwash'][0]['high']
                )
                # pos = self.np_random.uniform(self.dw_model.low, self.dw_model.high)
                # self.dw_model.update_pos(pos)
                randomize_flag = self.DISTURBANCES['downwash'][0]['randomize'] if 'randomize' in \
                                                                                  self.DISTURBANCES['downwash'][
                                                                                      0] else False
                if self.DISTURBANCES['downwash'][0]['mode'] == 'fix' and randomize_flag:
                    pos = self.np_random.uniform(self.dw_model.low, self.dw_model.high)
                    self.dw_model.update_pos(pos)
                    self.dw_model.reset()
                else:
                    pass  # randomization only implemented for fixed downwash
            elif self.DISTURBANCES['downwash'][0]['mode'] == 'track':
                self.dw_model = Downwash()  # update the position later

        # store the last prop values (only for logging)
        self.last_prop_values = None

    def set_goals(self):
        # Create X_GOAL and U_GOAL references for the assigned task.
        # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
        use_ilqr_ref = getattr(self, 'TASK_INFO', {}).get('ilqr_ref', False)
        use_pitch_ref = getattr(self, 'TASK_INFO', {}).get('pitch_ref', False)
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE,
                              QuadType.TWO_D_ATTITUDE_5S,
                              QuadType.TWO_D_ATTITUDE_BODY,]:
            self.U_GOAL = np.array([self.MASS * self.GRAVITY_ACC, 0.0])
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            self.U_GOAL = np.array([self.MASS * self.GRAVITY_ACC, 0.0, 0.0])
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            self.U_GOAL = np.array([self.MASS * self.GRAVITY_ACC, 0.0, 0.0, 0.0])
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            self.U_GOAL = np.array([self.MASS * self.GRAVITY_ACC, 0.0, 0.0, 0.0])
        else:
            self.U_GOAL = np.ones(self.action_dim) * self.MASS * self.GRAVITY_ACC / self.action_dim
        if self.TASK == Task.STABILIZATION:
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.hstack(
                    [self.TASK_INFO['stabilization_goal'][1],
                     0.0])  # x = {z, z_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta, theta_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta, theta_dot}.
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0, 0.0
                ])
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0, 0.0
                ])  # x = {x, x_dot, z, z_dot, theta}.
            elif self.QUAD_TYPE == QuadType.THREE_D:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0,
                    self.TASK_INFO['stabilization_goal'][2], 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])  # x = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0,
                    self.TASK_INFO['stabilization_goal'][2], 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                ])
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
                self.X_GOAL = np.hstack([
                    self.TASK_INFO['stabilization_goal'][0], 0.0,
                    self.TASK_INFO['stabilization_goal'][1], 0.0,
                    self.TASK_INFO['stabilization_goal'][2], 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, self.MASS * self.GRAVITY_ACC])

        elif self.TASK == Task.TRAJ_TRACKING:
            if isinstance(self.EPISODE_LEN_SEC, list):
                self.episode_len = self.np_random.choice(self.EPISODE_LEN_SEC)
            else:
                self.episode_len = self.EPISODE_LEN_SEC
            if use_ilqr_ref:
                assert self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE,
                                          QuadType.THREE_D_ATTITUDE], \
                    '[ERROR] in Quadrotor.set_goals(), iLQR reference only available for attitude control interface.'
                # if using iLQR or trajectory data generated by other controllers
                # NOTE: the reference is assumed to have the same sampling frequency
                # TODO: add support for different sampling frequency
                # traj_overhead_steps = 60
                if hasattr(self.TASK_INFO, 'ilqr_traj_data'):
                    traj_data = np.load(self.TASK_INFO['ilqr_traj_data'], allow_pickle=True).item()
                else:
                    traj_tag = '2D_attitude' if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE else '3D_attitude'
                    traj_data = np.load(os.path.join(script_dir,
                                                     '../../../benchmarking_sim/quadrotor/data',
                                                     f'ilqr_quadrotor_{traj_tag}_{self.episode_len}_ref_traj.npy'),
                                        allow_pickle=True).item()
                # max_ref_steps = self.PYB_FREQ * self.episode_len
                # ref_traj_steps = traj_data['obs'][0].shape[0]
                # if ref_traj_steps >= max_ref_steps + traj_overhead_steps:
                    # traj_data['obs'][0] = traj_data['obs'][0][:max_ref_steps+traj_overhead_steps]
                # if
                POS_REF = np.array(
                    [traj_data['obs'][:, 0], 0 * traj_data['obs'][:, 0], traj_data['obs'][:, 2]]).T
                VEL_REF = np.array(
                    [traj_data['obs'][:, 1], 0 * traj_data['obs'][:, 1], traj_data['obs'][:, 3]]).T
                PITCH_REF = np.array([traj_data['obs'][:, 4], traj_data['obs'][:, 5]]).T
            else:
                if hasattr(self.TASK_INFO, 'custom_snap_ref_traj'):
                    traj_data = np.load(self.TASK_INFO['custom_snap_ref_traj'], allow_pickle=True).item()
                    POS_REF = traj_data['POS_REF']
                    VEL_REF = traj_data['VEL_REF']
                else:
                    strings = self.TASK_INFO['strings'] if 'strings' in self.TASK_INFO else None
                    waypoints = self.TASK_INFO['waypoints'] if 'waypoints' in self.TASK_INFO else None
                    POS_REF, VEL_REF, _ = self._generate_trajectory(traj_type=self.TASK_INFO['trajectory_type'],
                                                                    traj_length=self.episode_len,
                                                                    num_cycles=self.TASK_INFO['num_cycles'],
                                                                    traj_plane=self.TASK_INFO['trajectory_plane'],
                                                                    position_offset=self.TASK_INFO[
                                                                        'trajectory_position_offset'],
                                                                    scaling=self.TASK_INFO['trajectory_scale'],
                                                                    sample_time=self.CTRL_TIMESTEP,
                                                                    string_list=strings,
                                                                    waypoint_list=waypoints
                                                                    )
                    CUSTOM_REF_TRAJ = {}
                    CUSTOM_REF_TRAJ['POS_REF'] = POS_REF
                    CUSTOM_REF_TRAJ['VEL_REF'] = VEL_REF
                    np.save(os.path.join(script_dir, '../../../benchmarking_sim/quadrotor/data',
                                         'custom_snap_ref_traj.npy'), CUSTOM_REF_TRAJ, allow_pickle=True)
                    # add attribute to self.TASK_INFO
                    self.TASK_INFO['custom_snap_ref_traj'] = os.path.join(script_dir, '../../../benchmarking_sim/quadrotor/data',
                                                                          'custom_snap_ref_traj.npy')
                    _plot_trajectory(POS_REF,
                                     waypoints=waypoints,
                                     strings=strings,
                                     save_path=os.path.join(script_dir, '../../../benchmarking_sim/quadrotor/data'))

                # Each of the 3 returned values is of shape (Ctrl timesteps, 3)
            if self.QUAD_TYPE == QuadType.ONE_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2]  # z_dot
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
                if use_ilqr_ref and use_pitch_ref:
                    self.X_GOAL = np.vstack([
                        POS_REF[:, 0],  # x
                        VEL_REF[:, 0],  # x_dot
                        POS_REF[:, 2],  # z
                        VEL_REF[:, 2],  # z_dot
                        PITCH_REF[:, 0],  # theta
                        PITCH_REF[:, 1],  # theta_dot
                    ]).transpose()
                else:
                    self.X_GOAL = np.vstack([
                        POS_REF[:, 0],  # x
                        VEL_REF[:, 0],  # x_dot
                        POS_REF[:, 2],  # z
                        VEL_REF[:, 2],  # z_dot
                        np.zeros(POS_REF.shape[0]),  # zeros
                        np.zeros(VEL_REF.shape[0])
                    ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D:
                # Additional transformation of the originally planar trajectory.
                POS_REF_TRANS, VEL_REF_TRANS = transform_trajectory(
                    POS_REF, VEL_REF, trans_info={
                        'point': self.TASK_INFO['proj_point'],
                        'normal': self.TASK_INFO['proj_normal'],
                    })
                self.X_GOAL = np.vstack([
                    POS_REF_TRANS[:, 0],  # x
                    VEL_REF_TRANS[:, 0],  # x_dot
                    POS_REF_TRANS[:, 1],  # y
                    VEL_REF_TRANS[:, 1],  # y_dot
                    POS_REF_TRANS[:, 2],  # z
                    VEL_REF_TRANS[:, 2],  # z_dot
                    np.zeros(POS_REF_TRANS.shape[0]),  # zeros
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(POS_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0]),
                    np.zeros(VEL_REF_TRANS.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 1],  # y
                    VEL_REF[:, 1],  # y_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 1],  # y
                    VEL_REF[:, 1],  # y_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0])
                ]).transpose()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
                self.X_GOAL = np.vstack([
                    POS_REF[:, 0],  # x
                    VEL_REF[:, 0],  # x_dot
                    POS_REF[:, 1],  # y
                    VEL_REF[:, 1],  # y_dot
                    POS_REF[:, 2],  # z
                    VEL_REF[:, 2],  # z_dot
                    np.zeros(POS_REF.shape[0]),  # zeros
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(POS_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]),
                    np.zeros(VEL_REF.shape[0]) * self.MASS * self.GRAVITY_ACC,
                ]).transpose()

    def reset(self, seed=None):
        """(Re-)initializes the environment to start an episode.

        Mandatory to call at least once after __init__().

        Args:
            seed (int): An optional seed to reseed the environment.

        Returns:
            obs (ndarray): The initial state of the environment.
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        """
        super().before_reset(seed=seed)
        # PyBullet simulation reset.
        super()._reset_simulation()

        # Choose randomized or deterministic inertial properties.
        prop_values = {
            'M': self.MASS,
            'Ixx': self.J[0, 0],
            'Iyy': self.J[1, 1],
            'Izz': self.J[2, 2]
        }
        if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            prop_values['beta_1'] = self.beta_1
            prop_values['beta_2'] = self.beta_2
            prop_values['alpha_1'] = self.alpha_1
            prop_values['alpha_2'] = self.alpha_2
            prop_values['alpha_3'] = self.alpha_3
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            prop_values['beta_1'] = self.beta_1
            prop_values['beta_2'] = self.beta_2
            prop_values['alpha_1'] = self.alpha_1
            prop_values['alpha_2'] = self.alpha_2
            prop_values['alpha_3'] = self.alpha_3
            prop_values['alpha_4'] = self.alpha_4
            prop_values['alpha_5'] = self.alpha_5
            prop_values['alpha_6'] = self.alpha_6
            prop_values['alpha_7'] = self.alpha_7
            prop_values['alpha_8'] = self.alpha_8
            prop_values['alpha_9'] = self.alpha_9
        if self.RANDOMIZED_INERTIAL_PROP:
            prop_values = self._randomize_values_by_info(prop_values, self.INERTIAL_PROP_RAND_INFO)
            if any(phy_quantity < 0 for phy_quantity in prop_values.values()):
                if self.QUAD_TYPE != QuadType.TWO_D_ATTITUDE and self.QUAD_TYPE != QuadType.THREE_D_ATTITUDE:
                    raise ValueError('[ERROR] in Quadrotor.reset(), negative randomized inertial properties.')
        self.OVERRIDDEN_QUAD_MASS = prop_values['M']
        self.OVERRIDDEN_QUAD_INERTIA = [prop_values['Ixx'], prop_values['Iyy'], prop_values['Izz']]
        if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            self.beta_1 = prop_values['beta_1']
            self.beta_2 = prop_values['beta_2']
            self.alpha_1 = prop_values['alpha_1']
            self.alpha_2 = prop_values['alpha_2']
            self.alpha_3 = prop_values['alpha_3']
            self._setup_symbolic(prop_values)
            self.setup_dynamics_si_expression(prop_values)
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            if self.PHYSICS == Physics.DYN_SI_3D:
                self.beta_1 = prop_values['beta_1']
                self.beta_2 = prop_values['beta_2']
                self.alpha_1 = prop_values['alpha_1']
                self.alpha_2 = prop_values['alpha_2']
                self.alpha_3 = prop_values['alpha_3']
                self.alpha_4 = prop_values['alpha_4']
                self.alpha_5 = prop_values['alpha_5']
                self.alpha_6 = prop_values['alpha_6']
                self.alpha_7 = prop_values['alpha_7']
                self.alpha_8 = prop_values['alpha_8']
                self.alpha_9 = prop_values['alpha_9']
                self._setup_symbolic(prop_values)
                self.setup_dynamics_si_3d_expression(prop_values)
        self.last_prop_values = prop_values

        # Override inertial properties.
        p.changeDynamics(
            self.DRONE_IDS[0],
            linkIndex=-1,  # Base link.
            mass=self.OVERRIDDEN_QUAD_MASS,
            localInertiaDiagonal=self.OVERRIDDEN_QUAD_INERTIA,
            physicsClientId=self.PYB_CLIENT)

        # Randomize initial state.
        init_values = {init_name: self.__dict__[init_name.upper()]
                       for init_name in self.INIT_STATE_LABELS[self.QUAD_TYPE]}
        if self.RANDOMIZED_INIT:
            init_values = self._randomize_values_by_info(init_values, self.INIT_STATE_RAND_INFO)
        INIT_XYZ = [init_values.get('init_' + k, 0.) for k in ['x', 'y', 'z']]
        INIT_VEL = [init_values.get('init_' + k + '_dot', 0.) for k in ['x', 'y', 'z']]
        INIT_RPY = [init_values.get('init_' + k, 0.) for k in ['phi', 'theta', 'psi']]
        if self.QUAD_TYPE == QuadType.TWO_D:
            INIT_ANG_VEL = [0, init_values.get('init_theta_dot', 0.), 0]
        # elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
        elif self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            INIT_ANG_VEL = [0, init_values.get('init_theta_dot', 0.), 0]
            self.attitude_control.reset()
        else:
            INIT_ANG_VEL = [init_values.get('init_' + k, 0.) for k in ['p', 'q', 'r']]  # TODO: transform from body rates.
        p.resetBasePositionAndOrientation(self.DRONE_IDS[0], INIT_XYZ,
                                          p.getQuaternionFromEuler(INIT_RPY),
                                          physicsClientId=self.PYB_CLIENT)
        p.resetBaseVelocity(self.DRONE_IDS[0], INIT_VEL, INIT_ANG_VEL,
                            physicsClientId=self.PYB_CLIENT)

        # Randomize disturbances.
        # disturbance_rand_dict = {}
        if self.RANDOMIZED_DISTURBANCE and self.DISTURBABCE_RAND_INFO is not None:
            # first reset the disturbances
            for keys, values in self.DISTURBANCES.items():
                if keys in self.disturbances:
                    if keys == 'downwash':
                        self.dw_model.update_pos(self.DISTURBANCES['downwash'][0]['pos'])
                    else:
                        # process and observation disturbances
                        self.disturbances[keys].disturbances[0].std = values[0]['std']

            # then randomize the disturbances #TODO: need a more elegant way to do this
            self.disturbance_before_rand = self.disturbances.copy()
            for keys, values in self.DISTURBABCE_RAND_INFO.items():
                if keys in self.disturbances:
                    if keys == 'downwash':
                        self.dw_model.update_pos(np.random.choice(values[0]['scale']))
                    else:
                        # process and observation disturbances
                        if isinstance(self.disturbances[keys].disturbances[0].std, list):
                            self.disturbances[keys].disturbances[0].std =\
                                list(np.array(self.disturbances[keys].disturbances[0].std) * np.random.choice(values[0]['scale']))
                        elif isinstance(self.disturbances[keys].disturbances[0].std, float):
                            self.disturbances[keys].disturbances[0].std =\
                                self.disturbances[keys].disturbances[0].std * np.random.choice(values[0]['scale'])

        # Update task goals
        # NOTE: Task info will be randomized when set_goals() is called.
        self.set_goals()

        # Update BaseAviary internal variables before calling self._get_observation().
        self._update_and_store_kinematic_information()
        obs, info = self._get_observation(), self._get_reset_info()
        obs, info = super().after_reset(obs, info)

        # reset disturbance model
        if 'downwash' in self.disturbances:
            randomize_flag = self.DISTURBANCES['downwash'][0]['randomize'] if 'randomize' in self.DISTURBANCES['downwash'][0] else False
            if self.DISTURBANCES['downwash'][0]['mode'] == 'fix' and randomize_flag:
                pos = self.np_random.uniform(self.dw_model.low, self.dw_model.high)
                self.dw_model.update_pos(pos)
                self.dw_model.reset()
            else:
                pass  # randomization only implemented for fixed downwash

        # Return either an observation and dictionary or just the observation.
        return obs, info

    def step(self, action):
        """Advances the environment by one control step.

        Pass the commanded RPMs and the adversarial force to the superclass .step().
        The PyBullet simulation is stepped PYB_FREQ/CTRL_FREQ times in BaseAviary.

        Args:
            action (ndarray): The action applied to the environment for the step.

        Returns:
            obs (ndarray): The state of the environment after the step.
            reward (float): The scalar reward/cost of the step.
            done (bool): Whether the conditions for the end of an episode are met in the step.
            info (dict): A dictionary with information about the constraints evaluations and violations.
        """

        # Get the preprocessed rpm for each motor
        action = super().before_step(action)

        # Determine disturbance force.
        disturb_force = None
        passive_disturb = 'dynamics' in self.disturbances
        adv_disturb = self.adversary_disturbance == 'dynamics'
        if 'downwash' in self.disturbances:
            disturb_force = np.zeros(self.DISTURBANCE_MODES['dynamics']['dim'])
        if passive_disturb or adv_disturb:
            disturb_force = np.zeros(self.DISTURBANCE_MODES['dynamics']['dim'])
        if passive_disturb:
            disturb_force = self.disturbances['dynamics'].apply(
                disturb_force, self)
        if adv_disturb and self.adv_action is not None:
            disturb_force = disturb_force + self.adv_action
            # Clear the adversary action, wait for the next one.
            self.adv_action = None
        # Construct full (3D) disturbance force.
        if disturb_force is not None:
            if self.QUAD_TYPE == QuadType.ONE_D:
                # Only disturb on z direction.
                disturb_force = [0, 0, float(disturb_force)]
            elif self.QUAD_TYPE == QuadType.TWO_D:
                # Only disturb on x-z plane.
                disturb_force = [float(disturb_force[0]), 0, float(disturb_force[1])]
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
                # Only disturb on x-z plane.
                disturb_force = [float(disturb_force[0]), 0, float(disturb_force[1])]
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
                # Only disturb on x-z plane.
                disturb_force = [float(disturb_force[0]), 0, float(disturb_force[1])]
            elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
                # Only disturb on x-z plane.
                disturb_force = [float(disturb_force[0]), 0, float(disturb_force[1])]
            elif self.QUAD_TYPE == QuadType.THREE_D:
                disturb_force = np.asarray(disturb_force).flatten()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
                disturb_force = np.asarray(disturb_force).flatten()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
                disturb_force = np.asarray(disturb_force).flatten()
            elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
                disturb_force = np.asarray(disturb_force).flatten()

        # handle downwash force
        if 'downwash' in self.disturbances:
            assert self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY, QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10, QuadType.THREE_D_ATTITUDE_DELAY], '[ERROR] in Quadrotor.step(), downwash model is only available for identified model.'
            # get the current observation
            # obs = self._get_observation()
            state = self.state
            # print(f'state: {state}')
            # print(f'get state: {self._get_state()}')
            # if self.DISTURBANCES['downwash'][0]['mode'] == 'track':
            #     # update the position of the downwash model
            #     self.dw_model.update_pos(pos=np.array([obs[0], 0, obs[2]]))
            # get the downwash force
            if self.QUAD_TYPE not in [QuadType.TWO_D_ATTITUDE]:
                raise ValueError('[ERROR] in Quadrotor.step(), downwash force is only available for 2D attitude model.')
            if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_BODY, QuadType.TWO_D_ATTITUDE_5S]:
                pos = np.array([state[0], 0, state[2]])
            elif self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10, QuadType.THREE_D_ATTITUDE_DELAY]:
                pos = np.array([state[0], state[2], state[4]])

            if self.DISTURBANCES['downwash'][0]['mode'] == 'track':
                # update the position of the downwash model
                self.dw_model.update_pos(pos=pos + self.DISTURBANCES['downwash'][0]['pos'])
            dw_force_mag = self.dw_model.get_dw_force_mag(target_pos=pos, mode='absolute')
            # print(f'dw_force_mag: {dw_force_mag:2f} [N]')

            # print(f'dw_force_mag: {dw_force_mag:2f} [N]')
            disturb_force[-1] += -dw_force_mag

        # Advance the simulation.
        super()._advance_simulation(action, disturb_force)
        # Standard Gym return.
        obs = self._get_observation()
        rew = self._get_reward()
        done = self._get_done()
        info = self._get_info()
        obs, rew, done, info = super().after_step(obs, rew, done, info)
        return obs, rew, done, info

    def render(self, mode='human', close=False):
        '''Retrieves a frame from PyBullet rendering.

        Args:
            mode (str): Unused.
            close (bool): Unused

        Returns:
            frame (ndarray): A multidimensional array with the RGB frame captured by PyBullet's camera.
        '''

        [w, h, rgb, _, _] = p.getCameraImage(width=self.RENDER_WIDTH,
                                             height=self.RENDER_HEIGHT,
                                             shadow=1,
                                             viewMatrix=self.CAM_VIEW,
                                             projectionMatrix=self.CAM_PRO,
                                             renderer=p.ER_TINY_RENDERER,
                                             flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                             physicsClientId=self.PYB_CLIENT)
        # Image.fromarray(np.reshape(rgb, (h, w, 4)), 'RGBA').show()
        return np.reshape(rgb, (h, w, 4))

    def _setup_symbolic(self, prior_prop={}, **kwargs):
        '''Creates symbolic (CasADi) models for dynamics, observation, and cost.

        Args:
            prior_prop (dict): specify the prior inertial prop to use in the symbolic model.
        '''
        m = prior_prop.get('M', self.MASS)
        Iyy = prior_prop.get('Iyy', self.J[1, 1])

        g, length = self.GRAVITY_ACC, self.L
        dt = self.CTRL_TIMESTEP
        # Define states.
        z = cs.MX.sym('z')
        z_dot = cs.MX.sym('z_dot')
        u_eq = m * g
        X_dot, parameterized_X_dot, Y = None, None, None
        lr_param = None  # external model parameters
        if self.QUAD_TYPE == QuadType.ONE_D:
            nx, nu = 2, 1
            # Define states.
            X = cs.vertcat(z, z_dot)
            # Define input thrust.
            T = cs.MX.sym('T')
            U = cs.vertcat(T)
            # Define dynamics equations.
            X_dot = cs.vertcat(z_dot, T / m - g)
            # Define observation equation.
            Y = cs.vertcat(z, z_dot)
        elif self.QUAD_TYPE == QuadType.TWO_D:
            nx, nu = 6, 2
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            theta = cs.MX.sym('theta')
            theta_dot = cs.MX.sym('theta_dot')
            X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
            # Define input thrusts.
            T1 = cs.MX.sym('T1')
            T2 = cs.MX.sym('T2')
            U = cs.vertcat(T1, T2)
            # Define dynamics equations.
            X_dot = cs.vertcat(x_dot,
                               cs.sin(theta) * (T1 + T2) / m,
                               z_dot,
                               cs.cos(theta) * (T1 + T2) / m - g,
                               theta_dot,
                               length * (T2 - T1) / Iyy / np.sqrt(2))
            # Define observation.
            Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            # identified parameters for the 2D attitude interface
            # NOTE: these parameters are not set in the prior_prop dict
            # since they are specific to the 2D attitude model
            self.beta_1 = prior_prop.get('beta_1', 17.7903)
            self.beta_2 = prior_prop.get('beta_2', 4.2209)
            self.beta_3 = prior_prop.get('beta_3', 0.0)
            self.alpha_1 = prior_prop.get('alpha_1', -149.740766)
            self.alpha_2 = prior_prop.get('alpha_2', -16.755920)
            self.alpha_3 = prior_prop.get('alpha_3', 129.041538)
            self.pitch_bias = prior_prop.get('pitch_bias', 0.0)

            nx, nu = 6, 2
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            theta_dot = cs.MX.sym('theta_dot')
            X = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
            # Define input collective thrust and theta.
            T = cs.MX.sym('T_c')  # normalized thrust [N]
            P = cs.MX.sym('P_c')  # desired pitch angle [rad]
            U = cs.vertcat(T, P)
            # The thrust in PWM is converted from the normalized thrust.
            # With the formulat F_desired = b_F * T + a_F

            # Define dynamics equations.
            # TODO: create a parameter for the new quad model
            # X_dot = cs.vertcat(x_dot,
            #                    (18.112984649321753 * T + 3.7613154938448576) * cs.sin(theta),
            #                    z_dot,
            #                    (18.112984649321753 * T + 3.7613154938448576) * cs.cos(theta) - g,
            #                    theta_dot,
            #                    # 60 * (60 * (P - theta) - theta_dot)
            #                    -143.9 * theta - 13.02 * theta_dot + 122.5 * P
            #                    )
            X_dot = cs.vertcat(x_dot,
                               (self.beta_1 * T + self.beta_2) * cs.sin(theta + self.pitch_bias) + self.beta_3,
                               z_dot,
                               (self.beta_1 * T + self.beta_2) * cs.cos(theta + self.pitch_bias) - g,
                               theta_dot,
                               self.alpha_1 * (theta + self.pitch_bias) + self.alpha_2 * theta_dot + self.alpha_3 * P)
            # Define observation.
            Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)

            T_mapping = self.beta_1 * T + self.beta_2
            P_mapping = self.alpha_1 * (theta + self.pitch_bias) + self.alpha_2 * theta_dot + self.alpha_3 * P
            self.T_mapping_func = cs.Function('T_mapping', [T], [T_mapping])
            self.P_mapping_func = cs.Function('P_mapping', [theta, theta_dot, P], [P_mapping])

            # mp = [18.112984649321753, 3.6800, 0.0, 140.8, 13.4, 124.8, 0.0]
            lr_param = cs.MX.sym('learnable_param', 5)
            parameterized_X_dot = cs.vertcat(
                x_dot,
                (lr_param[0] * self.beta_1 * T + lr_param[1] * self.beta_2) * cs.sin(theta),
                z_dot,
                (lr_param[0] * self.beta_1 * T + lr_param[1] * self.beta_2) * cs.cos(theta) - g,
                theta_dot,
                lr_param[2] * self.alpha_1 * theta + lr_param[3] * self.alpha_2 * theta_dot + lr_param[4] * self.alpha_3 * P
            )

        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
            nx, nu = 6, 2
            # Define states.
            x = cs.MX.sym('x')
            vx = cs.MX.sym('vx')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            theta_dot = cs.MX.sym('theta_dot')
            z = cs.MX.sym('z')
            vz = cs.MX.sym('vz')
            X = cs.vertcat(x, vx, z, vz, theta, theta_dot)
            # Define input collective thrust and theta.
            T = cs.MX.sym('T_c')  # normalized thrust [N]
            P = cs.MX.sym('P_c')  # desired pitch angle [rad]
            U = cs.vertcat(T, P)
            # The thrust in PWM is converted from the normalized thrust.
            # With the formulat F_desired = b_F * T + a_F

            # Define dynamics equations.
            X_dot = cs.vertcat(vx * cs.cos(theta) - vz * cs.sin(theta),
                               vz * theta_dot - g * cs.sin(theta),
                               vx * cs.sin(theta) + vz * cs.cos(theta),
                               -vx * theta_dot - g * cs.cos(theta) + (self.beta_1 * T + self.beta_2),
                               -theta_dot,
                               self.alpha_1 * (-theta + self.pitch_bias) + self.alpha_2 * -theta_dot + self.alpha_3 * P)
            # Define observation.
            x_dot = vx * cs.cos(theta) + vz * cs.sin(theta)
            z_dot = -vx * cs.sin(theta) + vz * cs.cos(theta)
            # Y = cs.vertcat(x, x_dot, z, z_dot, theta, theta_dot)
            Y = cs.vertcat(x, vx, z, vz, theta, theta_dot)
            T_mapping = self.beta_1 * T + self.beta_2
            self.T_mapping_func = cs.Function('T_mapping', [T], [T_mapping])

        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            nx, nu = 5, 2
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            X = cs.vertcat(x, x_dot, z, z_dot, theta)
            # Define input collective thrust and theta.
            T = cs.MX.sym('T_c')  # normalized thrust [N]
            P = cs.MX.sym('P_c')  # desired pitch angle [rad]
            U = cs.vertcat(T, P)
            # The thrust in PWM is converted from the normalized thrust.
            # With the formulat F_desired = b_F * T + a_F

            # Define dynamics equations.
            # TODO: create a parameter for the new quad model
            X_dot = cs.vertcat(x_dot,
                               (18.112984649321753 * T + 3.7613154938448576) * cs.sin(theta),
                               z_dot,
                               (18.112984649321753 * T + 3.7613154938448576) * cs.cos(theta) - g,
                               -60.00143727772195 * theta + 60.00143727772195 * P)
            # Define observation.
            Y = cs.vertcat(x, x_dot, z, z_dot, theta)
        elif self.QUAD_TYPE == QuadType.THREE_D:
            nx, nu = 12, 4
            Ixx = prior_prop.get('Ixx', self.J[0, 0])
            Izz = prior_prop.get('Izz', self.J[2, 2])
            J = cs.blockcat([[Ixx, 0.0, 0.0],
                             [0.0, Iyy, 0.0],
                             [0.0, 0.0, Izz]])
            Jinv = cs.blockcat([[1.0 / Ixx, 0.0, 0.0],
                                [0.0, 1.0 / Iyy, 0.0],
                                [0.0, 0.0, 1.0 / Izz]])
            gamma = self.KM / self.KF
            x = cs.MX.sym('x')
            y = cs.MX.sym('y')
            phi = cs.MX.sym('phi')  # Roll
            theta = cs.MX.sym('theta')  # Pitch
            psi = cs.MX.sym('psi')  # Yaw
            x_dot = cs.MX.sym('x_dot')
            y_dot = cs.MX.sym('y_dot')
            p_body = cs.MX.sym('p')  # Body frame roll rate
            q_body = cs.MX.sym('q')  # body frame pith rate
            r_body = cs.MX.sym('r')  # body frame yaw rate
            # PyBullet Euler angles use the SDFormat for rotation matrices.
            Rob = csRotXYZ(phi, theta, psi)  # rotation matrix transforming a vector in the body frame to the world frame.

            # Define state variables.
            X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)

            # Define inputs.
            f1 = cs.MX.sym('f1')
            f2 = cs.MX.sym('f2')
            f3 = cs.MX.sym('f3')
            f4 = cs.MX.sym('f4')
            U = cs.vertcat(f1, f2, f3, f4)

            # From Ch. 2 of Luis, Carlos, and Jérôme Le Ny. 'Design of a trajectory tracking controller for a
            # nanoquadcopter.' arXiv preprint arXiv:1608.05786 (2016).

            # Defining the dynamics function.
            # We are using the velocity of the base wrt to the world frame expressed in the world frame.
            # Note that the reference expresses this in the body frame.
            oVdot_cg_o = Rob @ cs.vertcat(0, 0, f1 + f2 + f3 + f4) / m - cs.vertcat(0, 0, g)
            pos_ddot = oVdot_cg_o
            pos_dot = cs.vertcat(x_dot, y_dot, z_dot)
            Mb = cs.vertcat(length / cs.sqrt(2.0) * (f1 + f2 - f3 - f4),
                            length / cs.sqrt(2.0) * (-f1 + f2 + f3 - f4),
                            gamma * (-f1 + f2 - f3 + f4))
            rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p_body, q_body, r_body)) @ J @ cs.vertcat(p_body, q_body, r_body)))
            ang_dot = cs.blockcat([[1, cs.sin(phi) * cs.tan(theta), cs.cos(phi) * cs.tan(theta)],
                                   [0, cs.cos(phi), -cs.sin(phi)],
                                   [0, cs.sin(phi) / cs.cos(theta), cs.cos(phi) / cs.cos(theta)]]) @ cs.vertcat(p_body, q_body, r_body)
            X_dot = cs.vertcat(pos_dot[0], pos_ddot[0], pos_dot[1], pos_ddot[1], pos_dot[2], pos_ddot[2], ang_dot, rate_dot)
            Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body)
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            nx, nu = 12, 4
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            y = cs.MX.sym('y')
            y_dot = cs.MX.sym('y_dot')
            phi = cs.MX.sym('phi')  # roll angle [rad]
            phi_dot = cs.MX.sym('phi_dot')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            theta_dot = cs.MX.sym('theta_dot')
            psi = cs.MX.sym('psi')  # yaw angle [rad]
            psi_dot = cs.MX.sym('psi_dot')
            X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot)
            # Define input collective thrust and theta.
            T = cs.MX.sym('T_c')  # normalized thrust [N]
            R = cs.MX.sym('R_c')  # desired roll angle [rad]
            P = cs.MX.sym('P_c')  # desired pitch angle [rad]
            Y_c = cs.MX.sym('Y_c')  # desired yaw angle [rad]
            U = cs.vertcat(T, R, P, Y_c)
            # The thrust in PWM is converted from the normalized thrust.
            # With the formulat F_desired = b_F * T + a_F
            # Haocheng's model
            # params_acc = [20.907574256269616, 3.653687545690674]
            # params_roll_rate = [-130.3, -16.33, 119.3]
            # params_pitch_rate = [-99.94, -13.3, 84.73]
            # params_yaw_rate = [0., 0., 0.]
            # Marcel's model
            # params_acc = [32.221212, 0.]
            # params_roll_rate = [-286.2, -23.03, 225.6]
            # params_pitch_rate = [-286.2, -23.03, 225.6]
            # params_yaw_rate = [-192.9, -22.22, 323.5]

            self.beta_1 = prior_prop.get('beta_1', 32.221212)
            self.beta_2 = prior_prop.get('beta_2', 0.)
            self.alpha_1 = prior_prop.get('alpha_1', -286.2)
            self.alpha_2 = prior_prop.get('alpha_2', -23.03)
            self.alpha_3 = prior_prop.get('alpha_3', 225.6)
            self.alpha_4 = prior_prop.get('alpha_4', -286.2)
            self.alpha_5 = prior_prop.get('alpha_5', -23.03)
            self.alpha_6 = prior_prop.get('alpha_6', 225.6)
            self.alpha_7 = prior_prop.get('alpha_7', -192.9)
            self.alpha_8 = prior_prop.get('alpha_8', -22.22)
            self.alpha_9 = prior_prop.get('alpha_9', 323.5)

            # Define dynamics equations.
            # TODO: create a parameter for the new quad model
            X_dot = cs.vertcat(x_dot,
                               (self.beta_1 * T + self.beta_2) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)),
                               y_dot,
                               (self.beta_1 * T + self.beta_2) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                               z_dot,
                               (self.beta_1 * T + self.beta_2) * cs.cos(phi) * cs.cos(theta) - g,
                               phi_dot,
                               theta_dot,
                               psi_dot,
                               self.alpha_1 * phi + self.alpha_2 * phi_dot + self.alpha_3 * R,
                               self.alpha_4 * theta + self.alpha_5 * theta_dot + self.alpha_6 * P,
                               self.alpha_7 * psi + self.alpha_8 * psi_dot + self.alpha_9 * Y_c)
            # Define observation.
            Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot)

            T_mapping = self.beta_1 * T + self.beta_2
            P_mapping = self.alpha_1 * theta + self.alpha_2 * theta_dot + self.alpha_3 * P
            R_mapping = self.alpha_4 * phi + self.alpha_5 * phi_dot + self.alpha_6 * R
            Y_mapping = self.alpha_7 * psi + self.alpha_8 * psi_dot + self.alpha_9 * Y_c
            self.T_mapping_func = cs.Function('T_mapping', [T], [T_mapping])
            self.P_mapping_func = cs.Function('P_mapping', [theta, theta_dot, P], [P_mapping])
            self.R_mapping_func = cs.Function('R_mapping', [phi, phi_dot, R], [R_mapping])
            self.Y_mapping_func = cs.Function('Y_mapping', [psi, psi_dot, Y_c], [Y_mapping])

        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            nx, nu = 10, 3
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            y = cs.MX.sym('y')
            y_dot = cs.MX.sym('y_dot')
            phi = cs.MX.sym('phi')  # roll angle [rad]
            phi_dot = cs.MX.sym('phi_dot')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            theta_dot = cs.MX.sym('theta_dot')
            X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, phi_dot, theta_dot)
            # Define input collective thrust and theta.
            T = cs.MX.sym('T_c')  # normalized thrust [N]
            R = cs.MX.sym('R_c')  # desired roll angle [rad]
            P = cs.MX.sym('P_c')  # desired pitch angle [rad]
            U = cs.vertcat(T, R, P)
            # The thrust in PWM is converted from the normalized thrust.
            # With the formulat F_desired = b_F * T + a_F
            params_acc = [20.907574256269616, 3.653687545690674]
            params_roll_rate = [-130.3, -16.33, 119.3]
            params_pitch_rate = [-99.94, -13.3, 84.73]
            psi = 0

            self.a = prior_prop.get('a', 20.907574256269616)
            self.b = prior_prop.get('b', 3.653687545690674)
            self.c = prior_prop.get('c', -130.3)
            self.d = prior_prop.get('d', -16.33)
            self.e = prior_prop.get('e', 119.3)
            self.f = prior_prop.get('f', -99.94)
            self.h = prior_prop.get('h', -13.3)
            self.ll = prior_prop.get('ll', 84.73)

            # Define dynamics equations.
            # TODO: create a parameter for the new quad model
            X_dot = cs.vertcat(x_dot,
                               (self.a * T + self.b) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)),
                               y_dot,
                               (self.a * T + self.b) * (
                                   cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                               z_dot,
                               (self.a * T + self.b) * cs.cos(phi) * cs.cos(theta) - g,
                               phi_dot,
                               theta_dot,
                               self.c * phi + self.d * phi_dot + self.e * R,
                               self.f * theta + self.h * theta_dot + self.ll * P,
                               )
            # Define observation.
            Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, phi_dot, theta_dot)

            T_mapping = self.a * T + self.b
            R_mapping = self.c * phi + self.d * phi_dot + self.e * R
            P_mapping = self.f * theta + self.h * theta_dot + self.ll * P
            self.T_mapping_func = cs.Function('T_mapping', [T], [T_mapping])
            self.P_mapping_func = cs.Function('P_mapping', [theta, theta_dot, P], [P_mapping])
            self.R_mapping_func = cs.Function('R_mapping', [phi, phi_dot, R], [R_mapping])

        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            nx, nu = 13, 4
            # Define states.
            x = cs.MX.sym('x')
            x_dot = cs.MX.sym('x_dot')
            y = cs.MX.sym('y')
            y_dot = cs.MX.sym('y_dot')
            phi = cs.MX.sym('phi')  # roll angle [rad]
            phi_dot = cs.MX.sym('phi_dot')
            theta = cs.MX.sym('theta')  # pitch angle [rad]
            theta_dot = cs.MX.sym('theta_dot')
            psi = cs.MX.sym('psi')  # yaw angle [rad]
            psi_dot = cs.MX.sym('psi_dot')
            thrust = cs.MX.sym('thrust')  # force from the motor
            X = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot, thrust)
            T_c = cs.MX.sym('T_c')  # normalized thrust [N]
            R_c = cs.MX.sym('R_c')  # desired roll angle [rad]
            P_c = cs.MX.sym('P_c')  # desired pitch angle [rad]
            Y_c = cs.MX.sym('Y_c')  # desired yaw angle [rad]
            U = cs.vertcat(T_c, R_c, P_c, Y_c)
            params_acc = prior_prop.get('param_acc', [1.0591, 0, 0.1108])
            params_roll_rate = prior_prop.get('params_roll_rate', [-286.2, -23.03, 225.6])
            params_pitch_rate = prior_prop.get('params_pitch_rate', [-286.2, -23.03, 225.6])
            params_yaw_rate = prior_prop.get('params_yaw_rate', [-192.9, -22.22, 323.5])
            thrust_dot = 1 / params_acc[2] * (T_c - thrust)  # [N/s]
            thrust_scaled = params_acc[0] * thrust + params_acc[1]  # [N]
            thrust_applied = thrust_scaled / 0.033
            # Define dynamics equations.
            X_dot = cs.vertcat(x_dot,
                               (thrust_applied) * (cs.cos(phi) * cs.sin(theta) * cs.cos(psi) + cs.sin(phi) * cs.sin(psi)),
                               y_dot,
                               (thrust_applied) * (cs.cos(phi) * cs.sin(theta) * cs.sin(psi) - cs.sin(phi) * cs.cos(psi)),
                               z_dot,
                               (thrust_applied) * cs.cos(phi) * cs.cos(theta) - g,
                               phi_dot,
                               theta_dot,
                               psi_dot,
                               params_roll_rate[0] * phi + params_roll_rate[1] * phi_dot + params_roll_rate[2] * R_c,
                               params_pitch_rate[0] * theta + params_pitch_rate[1] * theta_dot + params_pitch_rate[2] * P_c,
                               params_yaw_rate[0] * psi + params_yaw_rate[1] * psi_dot + params_yaw_rate[2] * Y_c,
                               thrust_dot)
            # Define observation.
            Y = cs.vertcat(x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, phi_dot, theta_dot, psi_dot, thrust_dot)

        # Set the equilibrium values for linearizations.
        X_EQ = np.zeros(self.state_dim)
        if self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE_DELAY]:
            X_EQ[-1] = self.MASS * g  # thrust equilibrium

        # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S]:
            U_EQ = np.array([u_eq, 0])
        elif self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_DELAY]:
            U_EQ = np.array([u_eq, 0, 0, 0])
        elif self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE_10]:
            U_EQ = np.array([u_eq, 0, 0])
        else:
            U_EQ = np.ones(self.action_dim) * u_eq / self.action_dim
        # Define cost (quadratic form).
        Q = cs.MX.sym('Q', nx, nx)
        R = cs.MX.sym('R', nu, nu)
        Xr = cs.MX.sym('Xr', nx, 1)
        Ur = cs.MX.sym('Ur', nu, 1)
        cost_func = 0.5 * (X - Xr).T @ Q @ (X - Xr) + 0.5 * (U - Ur).T @ R @ (U - Ur)
        # Define dynamics and cost dictionaries.
        dynamics = {
            'dyn_eqn': X_dot,
            'param_dyn_eqn': parameterized_X_dot,
            'obs_eqn': Y,
            'vars': {'X': X, 'U': U, 'P': lr_param}
        }
        cost = {
            'cost_func': cost_func,
            'vars': {
                'X': X,
                'U': U,
                'Xr': Xr,
                'Ur': Ur,
                'Q': Q,
                'R': R
            }
        }
        # Additional params to cache
        params = {
            # prior inertial properties
            'quad_mass': m,
            'quad_Iyy': Iyy,
            'quad_Ixx': Ixx if 'Ixx' in locals() else None,
            'quad_Izz': Izz if 'Izz' in locals() else None,
            # equilibrium point for linearization
            'X_EQ': X_EQ,
            'U_EQ': U_EQ,
        }
        # Setup symbolic model.
        self.symbolic = SymbolicModel(dynamics=dynamics, cost=cost, dt=dt, params=params)

    def _set_action_space(self):
        """Sets the action space of the environment."""
        # Define action/input dimension, labels, and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            action_dim = 1
            self.ACTION_LABELS = ['T']
            self.ACTION_UNITS = ['N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-']
        elif self.QUAD_TYPE == QuadType.TWO_D:
            action_dim = 2
            self.ACTION_LABELS = ['T1', 'T2']
            self.ACTION_UNITS = ['N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            action_dim = 2
            self.ACTION_LABELS = ['T_c', 'P_c']
            self.ACTION_UNITS = ['N', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            action_dim = 2
            self.ACTION_LABELS = ['T_c', 'P_c']
            self.ACTION_UNITS = ['N', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
            action_dim = 2
            self.ACTION_LABELS = ['T_c', 'P_c']
            self.ACTION_UNITS = ['N', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D:
            action_dim = 4
            self.ACTION_LABELS = ['T1', 'T2', 'T3', 'T4']
            self.ACTION_UNITS = ['N', 'N', 'N', 'N'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            action_dim = 4
            self.ACTION_LABELS = ['T_c', 'R_c', 'P_c', 'Y_c']
            self.ACTION_UNITS = ['N', 'rad', 'rad', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            action_dim = 3
            self.ACTION_LABELS = ['T_c', 'R_c', 'P_c']
            self.ACTION_UNITS = ['N', 'rad', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-']
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            action_dim = 4
            self.ACTION_LABELS = ['T_c', 'R_c', 'P_c', 'Y_c']
            self.ACTION_UNITS = ['N', 'rad', 'rad', 'rad'] if not self.NORMALIZED_RL_ACTION_SPACE else ['-', '-', '-', '-']
        # Defining physical bounds for actions
        max_roll_deg = 25
        max_pitch_deg = 25
        max_yaw_deg = 25
        max_roll_rad = max_roll_deg * math.pi / 180
        max_pitch_rad = max_pitch_deg * math.pi / 180
        max_yaw_rad = max_yaw_deg * math.pi / 180
        # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            n_mot = 4  # due to collective thrust
            a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            self.physical_action_bounds = (np.array([np.full(1, a_low, np.float32), np.full(1, -max_pitch_rad, np.float32)]).flatten(),
                                           np.array([np.full(1, a_high, np.float32), np.full(1, max_pitch_rad, np.float32)]).flatten())
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            n_mot = 4  # due to collective thrust
            # a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            # a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            a_low = 0.08  # [N] measured from hardware data
            a_high = 0.45  # [N]
            self.physical_action_bounds = (np.array([np.full(1, a_low, np.float32),
                                                     np.full(1, -max_roll_rad, np.float32),
                                                     np.full(1, -max_pitch_rad, np.float32),
                                                     np.full(1, -max_yaw_rad, np.float32)]).flatten(),
                                           np.array([np.full(1, a_high, np.float32),
                                                     np.full(1, max_roll_rad, np.float32),
                                                     np.full(1, max_pitch_rad, np.float32),
                                                     np.full(1, max_yaw_rad, np.float32)]).flatten())
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            n_mot = 4
            a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            self.physical_action_bounds = (np.array([np.full(1, a_low, np.float32),
                                                     np.full(1, -max_roll_rad, np.float32),
                                                     np.full(1, -max_pitch_rad, np.float32)]).flatten(),
                                           np.array([np.full(1, a_high, np.float32),
                                                     np.full(1, max_roll_rad, np.float32),
                                                     np.full(1, max_pitch_rad, np.float32)]).flatten())
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            n_mot = 4
            # a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            # a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            a_low = 0.08  # [N] measured from hardware data
            a_high = 0.45
            self.physical_action_bounds = (np.array([np.full(1, a_low, np.float32),
                                                     np.full(1, -max_roll_rad, np.float32),
                                                     np.full(1, -max_pitch_rad, np.float32),
                                                     np.full(1, -max_yaw_rad, np.float32)]).flatten(),
                                           np.array([np.full(1, a_high, np.float32),
                                                     np.full(1, max_roll_rad, np.float32),
                                                     np.full(1, max_pitch_rad, np.float32),
                                                     np.full(1, max_yaw_rad, np.float32)]).flatten())

        else:
            n_mot = 4 / action_dim
            a_low = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MIN_PWM + self.PWM2RPM_CONST)**2
            a_high = self.KF * n_mot * (self.PWM2RPM_SCALE * self.MAX_PWM + self.PWM2RPM_CONST)**2
            self.physical_action_bounds = (np.full(action_dim, a_low, np.float32),
                                           np.full(action_dim, a_high, np.float32))

        if self.NORMALIZED_RL_ACTION_SPACE:
            # Normalized thrust (around hover thrust).
            # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
                self.hover_thrust = self.GRAVITY_ACC * self.MASS
            else:
                self.hover_thrust = self.GRAVITY_ACC * self.MASS / action_dim

            self.action_scale = (self.physical_action_bounds[1] - self.physical_action_bounds[0]) / 2
            self.action_bias = (self.physical_action_bounds[1] + self.physical_action_bounds[0]) / 2
            self.action_space = spaces.Box(low=-np.ones(action_dim),
                                           high=np.ones(action_dim),
                                           dtype=np.float32)
        else:
            # Direct thrust control.
            self.action_space = spaces.Box(low=self.physical_action_bounds[0],
                                           high=self.physical_action_bounds[1],
                                           dtype=np.float32)

    def _set_observation_space(self):
        """Sets the observation space of the environment."""
        self.x_threshold = 2
        self.y_threshold = 2
        self.z_threshold = 2
        self.phi_threshold_radians = 85 * math.pi / 180
        self.theta_threshold_radians = 85 * math.pi / 180
        self.psi_threshold_radians = 180 * math.pi / 180  # Do not bound yaw.
        self.x_dot_threshold = 15
        self.y_dot_threshold = 15
        self.z_dot_threshold = 15
        self.phi_dot_threshold_radians = 500 * math.pi / 180
        self.theta_dot_threshold_radians = 500 * math.pi / 180
        self.psi_dot_threshold_radians = 500 * math.pi / 180
        self.force_motor_low = 0.08  # [N] measured from hardware data
        self.force_motor_high = 0.45  # [N] measured from hardware data

        if self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10, QuadType.THREE_D_ATTITUDE_DELAY]:
            # space for obstacle course
            self.x_threshold = 3
            self.y_threshold = 2
            self.z_threshold = 2.5

        # Define obs/state bounds, labels and units.
        if self.QUAD_TYPE == QuadType.ONE_D:
            # obs/state = {z, z_dot}.
            low = np.array([self.GROUND_PLANE_Z, -self.z_dot_threshold])
            high = np.array([self.z_threshold, self.z_dot_threshold])
            self.STATE_LABELS = ['z', 'z_dot']
            self.STATE_UNITS = ['m', 'm/s']
        # elif self.QUAD_TYPE == QuadType.TWO_D or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        elif self.QUAD_TYPE in [QuadType.TWO_D, QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_BODY]:
            # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
            low = np.array([
                -self.x_threshold, -self.x_dot_threshold,
                self.GROUND_PLANE_Z, -self.z_dot_threshold,
                -self.theta_threshold_radians, -self.theta_dot_threshold_radians
            ])
            high = np.array([
                self.x_threshold, self.x_dot_threshold,
                self.z_threshold, self.z_dot_threshold,
                self.theta_threshold_radians, self.theta_dot_threshold_radians
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta', 'theta_dot']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad', 'rad/s']
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            # obs/state = {x, x_dot, z, z_dot, theta, theta_dot}.
            low = np.array([
                -self.x_threshold, -self.x_dot_threshold,
                self.GROUND_PLANE_Z, -self.z_dot_threshold,
                -self.theta_threshold_radians
            ])
            high = np.array([
                self.x_threshold, self.x_dot_threshold,
                self.z_threshold, self.z_dot_threshold,
                self.theta_threshold_radians
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'z', 'z_dot', 'theta']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'rad']
        elif self.QUAD_TYPE == QuadType.THREE_D or self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            # obs/state = {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            low = np.array([
                -self.x_threshold, -self.x_dot_threshold,
                -self.y_threshold, -self.y_dot_threshold,
                self.GROUND_PLANE_Z, -self.z_dot_threshold,
                -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
                -self.phi_dot_threshold_radians, -self.theta_dot_threshold_radians, -self.psi_dot_threshold_radians
            ])
            high = np.array([
                self.x_threshold, self.x_dot_threshold,
                self.y_threshold, self.y_dot_threshold,
                self.z_threshold, self.z_dot_threshold,
                self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
                self.phi_dot_threshold_radians, self.theta_dot_threshold_radians, self.psi_dot_threshold_radians
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'psi', 'p', 'q', 'r']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s']
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            low = np.array([
                -self.x_threshold, -self.x_dot_threshold,
                -self.y_threshold, -self.y_dot_threshold,
                self.GROUND_PLANE_Z, -self.z_dot_threshold,
                -self.phi_threshold_radians, -self.theta_threshold_radians,
                -self.phi_dot_threshold_radians, -self.theta_dot_threshold_radians
            ])
            high = np.array([
                self.x_threshold, self.x_dot_threshold,
                self.y_threshold, self.y_dot_threshold,
                self.z_threshold, self.z_dot_threshold,
                self.phi_threshold_radians, self.theta_threshold_radians,
                self.phi_dot_threshold_radians, self.theta_dot_threshold_radians
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'phi_dot', 'theta_dot']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad/s', 'rad/s']
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            low = np.array([
                -self.x_threshold, -self.x_dot_threshold,
                -self.y_threshold, -self.y_dot_threshold,
                self.GROUND_PLANE_Z, -self.z_dot_threshold,
                -self.phi_threshold_radians, -self.theta_threshold_radians, -self.psi_threshold_radians,
                -self.phi_dot_threshold_radians, -self.theta_dot_threshold_radians, -self.psi_dot_threshold_radians,
                self.force_motor_low
            ])
            high = np.array([
                self.x_threshold, self.x_dot_threshold,
                self.y_threshold, self.y_dot_threshold,
                self.z_threshold, self.z_dot_threshold,
                self.phi_threshold_radians, self.theta_threshold_radians, self.psi_threshold_radians,
                self.phi_dot_threshold_radians, self.theta_dot_threshold_radians, self.psi_dot_threshold_radians,
                self.force_motor_high
            ])
            self.STATE_LABELS = ['x', 'x_dot', 'y', 'y_dot', 'z', 'z_dot',
                                 'phi', 'theta', 'psi', 'phi_dot', 'theta_dot', 'psi_dot', 'force_motor']
            self.STATE_UNITS = ['m', 'm/s', 'm', 'm/s', 'm', 'm/s',
                                'rad', 'rad', 'rad', 'rad/s', 'rad/s', 'rad/s', 'N']

        # Define the state space for the dynamics.
        self.state_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Concatenate reference for RL.
        if self.COST == Cost.RL_REWARD and self.TASK == Task.TRAJ_TRACKING and self.obs_goal_horizon > 0:
            # Include future goal state(s).
            # e.g. horizon=1, obs = {state, state_target}
            mul = 1 + self.obs_goal_horizon
            low = np.concatenate([low] * mul)
            high = np.concatenate([high] * mul)
        elif self.COST == Cost.RL_REWARD and self.TASK == Task.STABILIZATION and self.obs_goal_horizon > 0:
            low = np.concatenate([low] * 2)
            high = np.concatenate([high] * 2)

        # Define obs space exposed to the controller.
        # Note how the obs space can differ from state space (i.e. augmented with the next reference states for RL)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _setup_disturbances(self):
        """Sets up the disturbances."""
        # Custom disturbance info.
        self.DISTURBANCE_MODES['observation']['dim'] = self.obs_dim
        self.DISTURBANCE_MODES['action']['dim'] = self.action_dim
        self.DISTURBANCE_MODES['dynamics']['dim'] = int(self.QUAD_TYPE)
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            self.DISTURBANCE_MODES['dynamics']['dim'] = 2
        elif self.QUAD_TYPE in [QuadType.THREE_D_ATTITUDE, QuadType.THREE_D_ATTITUDE_10, QuadType.THREE_D_ATTITUDE_DELAY]:
            self.DISTURBANCE_MODES['dynamics']['dim'] = 3
        self.DISTURBANCE_MODES['downwash']['dim'] = 3
        super()._setup_disturbances()

    # noinspection PyUnreachableCode
    def _preprocess_control(self, action):
        """Converts the action passed to .step() into motors' RPMs (ndarray of shape (4,)).

        Args:
            action (ndarray): The raw action input, of size 1 or 2 depending on QUAD_TYPE.

        Returns:
            action (ndarray): The motors RPMs to apply to the quadrotor.
        """
        action = self.denormalize_action(action)
        # self.current_physical_action = self.normalize_action(action)
        self.current_physical_action = action

        # Apply disturbances.
        if 'action' in self.disturbances:
            self.current_physical_action = self.disturbances['action'].apply(self.current_physical_action, self)
        if self.adversary_disturbance == 'action':
            self.current_physical_action = self.current_physical_action + self.adv_action
        self.current_noisy_physical_action = self.current_physical_action

        # Identified dynamics model works with collective thrust and pitch directly
        # No need to compute RPMs, (save compute)
        self.current_clipped_action = np.clip(self.current_noisy_physical_action,
                                              self.action_space.low,
                                              self.action_space.high)

        # TODO: double check why a mixture of PHYSICS and QUAD_TYPE is used here
        if self.PHYSICS in [Physics.DYN_SI, Physics.DYN_SI_3D, Physics.DYN_SI_3D_10, Physics.DYN_SI_3D_DELAY]:
            return self.current_clipped_action

        # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
        if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            collective_thrust, pitch = self.current_clipped_action

            if self.PHYSICS == Physics.DYN_2D:
                quat = get_quaternion_from_euler(self.rpy[0, :])
            else:
                _, quat = p.getBasePositionAndOrientation(self.DRONE_IDS[0], physicsClientId=self.PYB_CLIENT)
            thrust_action = self.attitude_control._dslPIDAttitudeControl(collective_thrust / 4,
                                                                         quat, np.array([0, pitch, 0]))
            # input thrust is in Newton
            thrust = np.array([thrust_action[0] + thrust_action[3], thrust_action[1] + thrust_action[2]])
            thrust = np.clip(thrust, np.full(2, self.physical_action_bounds[0][0] / 2),
                             np.full(2, self.physical_action_bounds[1][0] / 2))
            pitch = np.clip(pitch, self.physical_action_bounds[0][1], self.physical_action_bounds[1][1])
            self.current_clipped_action = np.array([sum(thrust), pitch])
        else:
            thrust = np.clip(action, self.physical_action_bounds[0], self.physical_action_bounds[1])
            self.current_clipped_action = thrust
        # convert to quad motor rpm commands
        pwm = cmd2pwm(thrust, self.PWM2RPM_SCALE, self.PWM2RPM_CONST, self.KF, self.MIN_PWM, self.MAX_PWM)
        rpm = pwm2rpm(pwm, self.PWM2RPM_SCALE, self.PWM2RPM_CONST)
        return rpm

    def normalize_action(self, action):
        """Converts a physical action into a normalized action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            normalized_action (ndarray): The action in the correct action space.
        """
        if self.NORMALIZED_RL_ACTION_SPACE:
            # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            # if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            #     action = np.array([(action[0] / self.hover_thrust - 1) / self.norm_act_scale, action[1]])
            # else:
            #     action = (action / self.hover_thrust - 1) / self.norm_act_scale

            action = (action - self.action_bias) / self.action_scale

        return action

    def denormalize_action(self, action):
        """Converts a normalized action into a physical action if necessary.

        Args:
            action (ndarray): The action to be converted.

        Returns:
            physical_action (ndarray): The physical action.
        """
        if self.NORMALIZED_RL_ACTION_SPACE:
            # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE or self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            # if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
            #     # # divided by 4 as action[0] is a collective thrust
            #     # thrust = action[0] / 4
            #     # hover_pwm = (self.HOVER_RPM - self.PWM2RPM_CONST) / self.PWM2RPM_SCALE
            #     # thrust = np.where(thrust <= 0, self.MIN_PWM + (thrust + 1) * (hover_pwm - self.MIN_PWM),
            #     #                    hover_pwm + (self.MAX_PWM - hover_pwm) * thrust)
            #
            #     thrust = (1 + self.norm_act_scale * action[0]) * self.hover_thrust
            #     # thrust = self.attitude_control.thrust2pwm(thrust)
            #     # thrust = self.HOVER_RPM * (1+0.05*action[0])
            #
            #     action = np.array([thrust, action[1]])
            # else:
            #     action = (1 + self.norm_act_scale * action) * self.hover_thrust

            action = action * self.action_scale + self.action_bias
            # action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def _get_state(self):
        """Returns the current state of the environment.

        Returns:
            obs (ndarray): The state of the quadrotor, of size 2 or 6 depending on QUAD_TYPE.
        """
        full_state = self._get_drone_state_vector(0)
        pos, _, rpy, vel, ang_v, rpy_rate, _ = np.split(full_state, [3, 7, 10, 13, 16, 19])
        if self.QUAD_TYPE == QuadType.ONE_D:
            # {z, z_dot}.
            self.state = np.hstack([pos[2], vel[2]]).reshape((2,))
        elif self.QUAD_TYPE == QuadType.TWO_D:
            # {x, x_dot, z, z_dot, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]
            ).reshape((6,))
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            # {x, x_dot, z, z_dot, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[2], vel[2], rpy[1], rpy_rate[1]]
            ).reshape((6,))
        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
            # perform transformation to body frame translational velocities
            pitch = -rpy[1]
            vx = vel[0] * np.cos(pitch) - vel[2] * np.sin(pitch)
            vz = vel[0] * np.sin(pitch) + vel[2] * np.cos(pitch)
            # {x, vx, z, vz, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vx, pos[2], vz, pitch, rpy_rate[1]]
            ).reshape((6,))

        elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
            # {x, x_dot, z, z_dot, theta, theta_dot}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[2], vel[2], rpy[1]]
            ).reshape((5,))
        elif self.QUAD_TYPE == QuadType.THREE_D:
            Rob = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape((3, 3))
            Rbo = Rob.T
            ang_v_body_frame = Rbo @ ang_v
            # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            self.state = np.hstack(
                # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]  # Note: world ang_v != body frame pqr
                [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v_body_frame]
            ).reshape((12,))
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
            # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            self.state = np.hstack(
                # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]
                [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, rpy_rate]
            ).reshape((12,))
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
            # {x, x_dot, y, y_dot, z, z_dot, phi, theta, p_body, q_body}.
            self.state = np.hstack(
                [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy[0], rpy[1], ang_v[0], ang_v[1]]
            ).reshape((10,))
        elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
            # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
            force_motor = self.current_physical_action[0] if self.current_physical_action is not None else 0.0
            self.state = np.hstack(
                [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy[0], rpy[1], rpy[2],
                 ang_v[0], ang_v[1], ang_v[2], force_motor]
            ).reshape((13,))
        # if not np.array_equal(self.state,
        #                       np.clip(self.state, self.observation_space.low, self.observation_space.high)):
        #     if self.GUI and self.VERBOSE:
        #         print(
        #             '[WARNING]: observation was clipped in Quadrotor._get_observation().'
        #         )
        return self.state

    def _get_observation(self):
        """Returns the current observation (state) of the environment.

        Returns:
            obs (ndarray): The state of the quadrotor, of size 2 or 6 depending on QUAD_TYPE.
        """
        # full_state = self._get_drone_state_vector(0)
        # pos, _, rpy, vel, ang_v, rpy_rate, _ = np.split(full_state, [3, 7, 10, 13, 16, 19])
        # if self.QUAD_TYPE == QuadType.ONE_D:
        #     # {z, z_dot}.
        #     self.state = np.hstack([pos[2], vel[2]]).reshape((2,))
        # elif self.QUAD_TYPE == QuadType.TWO_D:
        #     # {x, x_dot, z, z_dot, theta, theta_dot}.
        #     self.state = np.hstack(
        #         [pos[0], vel[0], pos[2], vel[2], rpy[1], ang_v[1]]
        #     ).reshape((6,))
        # elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
        #     # {x, x_dot, z, z_dot, theta, theta_dot}.
        #     self.state = np.hstack(
        #         [pos[0], vel[0], pos[2], vel[2], rpy[1], rpy_rate[1]]
        #     ).reshape((6,))
        # elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_BODY:
        #     # perform transformation to body frame translational velocities
        #     pitch = -rpy[1]
        #     vx = vel[0] * np.cos(pitch) - vel[2] * np.sin(pitch)
        #     vz = vel[0] * np.sin(pitch) + vel[2] * np.cos(pitch)
        #     # {x, vx, z, vz, theta, theta_dot}.
        #     self.state = np.hstack(
        #         [pos[0], vx, pos[2], vz, pitch, rpy_rate[1]]
        #     ).reshape((6,))
        #     world_state = np.hstack(
        #         [pos[0], vel[0], pos[2], vel[2], rpy[1], rpy_rate[1]]
        #     ).reshape((6,))
        #     print('world_state: ', world_state)
        #
        # elif self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
        #     # {x, x_dot, z, z_dot, theta, theta_dot}.
        #     self.state = np.hstack(
        #         [pos[0], vel[0], pos[2], vel[2], rpy[1]]
        #     ).reshape((5,))
        # elif self.QUAD_TYPE == QuadType.THREE_D:
        #     Rob = np.array(p.getMatrixFromQuaternion(self.quat[0])).reshape((3, 3))
        #     Rbo = Rob.T
        #     ang_v_body_frame = Rbo @ ang_v
        #     # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        #     self.state = np.hstack(
        #         # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]  # Note: world ang_v != body frame pqr
        #         [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v_body_frame]
        #     ).reshape((12,))
        # elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
        #     # {x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p_body, q_body, r_body}.
        #     self.state = np.hstack(
        #         # [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]
        #         [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy, ang_v]
        #     ).reshape((12,))
        # elif self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
        #     # {x, x_dot, y, y_dot, z, z_dot, phi, theta, p_body, q_body}.
        #     self.state = np.hstack(
        #         [pos[0], vel[0], pos[1], vel[1], pos[2], vel[2], rpy[0], rpy[1], ang_v[0], ang_v[1]]
        #     ).reshape((10,))
        # if not np.array_equal(self.state,
        #                       np.clip(self.state, self.observation_space.low, self.observation_space.high)):
        #     if self.GUI and self.VERBOSE:
        #         print(
        #             '[WARNING]: observation was clipped in Quadrotor._get_observation().'
        #         )
        _ = self._get_state()

        # Concatenate goal info (references state(s)) for RL.
        # Plus two because ctrl_step_counter has not incremented yet, and we want to return the obs (which would be
        # ctrl_step_counter + 1 as the action has already been applied), and the next state (+ 2) for the RL to see
        # the next state.
        obs = deepcopy(self.state)
        if self.at_reset:
            self.obs = self.extend_obs(obs, 1)
        else:
            self.obs = self.extend_obs(obs, self.ctrl_step_counter + 2)
        # Apply observation disturbance.
        if 'observation' in self.disturbances:
            self.obs = self.disturbances['observation'].apply(self.obs, self)
        return self.obs

    def _get_reward(self):
        """Computes the current step's reward value.

        Returns:
            reward (float): The evaluated reward/cost.
        """
        obs = self.shrink_obs(self.obs, self.ctrl_step_counter + 2)
        # obs = self.state
        # RL cost.
        if self.COST == Cost.RL_REWARD:
            act = np.asarray(self.current_clipped_action)
            act_error = act - self.U_GOAL
            # Quadratic costs w.r.t state and action
            # TODO: consider using multiple future goal states for cost in tracking
            if self.TASK == Task.STABILIZATION:
                state_error = obs - self.X_GOAL
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act_error * act_error)
            if self.TASK == Task.TRAJ_TRACKING:
                wp_idx = min(self.ctrl_step_counter + 1, self.X_GOAL.shape[
                    0] - 1)  # +1 because state has already advanced but counter not incremented.
                state_error = obs - self.X_GOAL[wp_idx]
                dist = np.sum(self.rew_state_weight * state_error * state_error)
                dist += np.sum(self.rew_act_weight * act_error * act_error)
            rew = -dist
            # Convert rew to be positive and bounded [0,1].
            if self.rew_exponential:
                rew = np.exp(rew)
            return rew

        # Control cost.
        if self.use_rl_reward_weights:
            self.Q = np.diag(self.rew_state_weight)
            self.R = np.diag(self.rew_act_weight)
        if self.COST == Cost.QUADRATIC:
            if self.TASK == Task.STABILIZATION:
                return float(-1 * self.symbolic.loss(x=obs,
                                                     Xr=self.X_GOAL,
                                                     u=self.current_clipped_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)['l'])
            if self.TASK == Task.TRAJ_TRACKING:
                return float(-1 * self.symbolic.loss(x=obs,
                                                     Xr=self.X_GOAL[self.ctrl_step_counter + 1, :],
                                                     # +1 because state has already advanced but counter not incremented.
                                                     u=self.current_clipped_action,
                                                     Ur=self.U_GOAL,
                                                     Q=self.Q,
                                                     R=self.R)['l'])
        return None

    def _get_done(self):
        """Computes the conditions for termination of an episode.

        Returns:
            done (bool): Whether an episode is over.
        """
        # Done if goal reached for stabilization task with quadratic cost.
        if self.TASK == Task.STABILIZATION:
            self.goal_reached = bool(
                np.linalg.norm(self.state - self.X_GOAL) < self.TASK_INFO['stabilization_goal_tolerance'])
            if self.goal_reached:
                return True

        # Done if state is out-of-bounds.
        if self.done_on_out_of_bound:
            if self.QUAD_TYPE == QuadType.ONE_D:
                mask = np.array([1, 0])
            if self.QUAD_TYPE == QuadType.TWO_D:
                mask = np.array([1, 0, 1, 0, 1, 0])
            # if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE:
            if self.QUAD_TYPE in [QuadType.TWO_D_ATTITUDE, QuadType.TWO_D_ATTITUDE_5S, QuadType.TWO_D_ATTITUDE_BODY]:
                mask = np.array([1, 0, 1, 0, 1, 0])
            if self.QUAD_TYPE == QuadType.TWO_D_ATTITUDE_5S:
                mask = np.array([1, 0, 1, 0, 1])
            if self.QUAD_TYPE == QuadType.THREE_D:
                mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
            if self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE:
                mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0])
            if self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_10:
                mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 0])
            if self.QUAD_TYPE == QuadType.THREE_D_ATTITUDE_DELAY:
                mask = np.array([1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0])
            # Element-wise or to check out-of-bound conditions.
            self.out_of_bounds = np.logical_or(self.state < self.state_space.low,
                                               self.state > self.state_space.high)
            # Mask out un-included dimensions (i.e. velocities)
            self.out_of_bounds = np.any(self.out_of_bounds * mask)
            # Early terminate if needed.
            if self.out_of_bounds:
                return True
        self.out_of_bounds = False

        return False

    def _get_info(self):
        """Generates the info dictionary returned by every call to .step().

        Returns:
            info (dict): A dictionary with information about the constraints evaluations and violations.
        """
        info = {}
        if self.TASK == Task.STABILIZATION and self.COST == Cost.QUADRATIC:
            info['goal_reached'] = self.goal_reached  # Add boolean flag for the goal being reached.
        if self.done_on_out_of_bound:
            info['out_of_bounds'] = self.out_of_bounds
        # Add MSE.
        state = deepcopy(self.state)
        if self.TASK == Task.STABILIZATION:
            state_error = state - self.X_GOAL
        elif self.TASK == Task.TRAJ_TRACKING:
            # TODO: should use angle wrapping
            # state[4] = normalize_angle(state[4])
            wp_idx = min(self.ctrl_step_counter + 1,
                         self.X_GOAL.shape[0] - 1)  # +1 so that state is being compared with proper reference state.
            state_error = state - self.X_GOAL[wp_idx]
        # Filter only relevant dimensions.
        state_error = state_error * self.info_mse_metric_state_weight
        info['mse'] = np.sum(state_error ** 2)
        if self.constraints is not None:
            info['constraint_values'] = self.constraints.get_values(self)
            info['constraint_violations'] = self.constraints.get_violations(self)
        return info

    def _get_reset_info(self):
        """Generates the info dictionary returned by every call to .reset().

        Returns:
            info (dict): A dictionary with information about the dynamics and constraints symbolic models.
        """
        info = {'symbolic_model': self.symbolic, 'physical_parameters': {
            'quadrotor_mass': self.OVERRIDDEN_QUAD_MASS,
            'quadrotor_inertia': self.OVERRIDDEN_QUAD_INERTIA,
        }, 'x_reference': self.X_GOAL, 'u_reference': self.U_GOAL}
        if self.constraints is not None:
            info['symbolic_constraints'] = self.constraints.get_all_symbolic_models()
        return info
