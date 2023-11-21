import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

from functools import partial

import numpy as np

from experiments.crazyflie.crazyflie_utils import gen_traj
from safe_control_gym.controllers.firmware.firmware_wrapper import Command
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.registration import make

save_data = True
TEST = 0
CERTIFY = True
COST_FUNCTION = 'one_step_cost'
M = 2

algo = 'lqr'
task = 'quadrotor'
sf = 'nl_mpsc'

algo_config = {
    'q_lqr': [0.1],
    'r_lqr': [0.1],
    'discrete_dynamics': True
}
task_config = {
    'info_in_reset': True,
    'ctrl_freq': 25,
    'pyb_freq': 1000,
    'physics': 'pyb',
    'gui': False,
    'quad_type': 3,
    'normalized_rl_action_space': False,
    'episode_len_sec': 20,
    'inertial_prop':
    {'M': 0.03775,
     'Ixx': 1.4e-05,
     'Iyy': 1.4e-05,
     'Izz': 2.17e-05},
    'randomized_inertial_prop': False,
    'task': 'stabilization',
    'task_info':
    {'stabilization_goal': [0.5, -0.5, 2],
     'stabilization_goal_tolerance': 0.0},
    'cost': 'quadratic',
    'constraints':
    [
        {'constraint_form': 'default_constraint',
         'constrained_variable': 'state',
         'upper_bounds': [0.75, 0.5, 1, 1, 2, 1, 0.2, 0.2, 0.2, 1, 1, 1],
         'lower_bounds': [-0.75, -0.5, -1, -1, 0, -1, -0.2, -0.2, -0.2, -1, -1, -1]},
        {'constraint_form': 'default_constraint',
         'constrained_variable': 'input',
         }
    ],
    'done_on_violation': False,
    'done_on_out_of_bound': True,
    'seed': 1337,
}
sf_config = {
    'r_lin': [90],
    'q_lin': [0.001, 0.06, 0.001, 0.06, 0.00025, 80, 1e-05, 1e-05, 0.75, 1, 1, 1],
    'horizon': 10,
    'warmstart': True,
    'integration_algo': 'rk4',
    'use_terminal_set': True,
    'cost_function': COST_FUNCTION,
    'mpsc_cost_horizon': M,
    'decay_factor': 0.85,
    'prior_info': {
        'prior_prop': None,
        'randomize_prior_prop': False,
        'prior_prop_rand_info': None},
    'n_samples': 600
}


class Controller():
    '''Template controller class. '''

    def __init__(self,
                 initial_obs,
                 initial_info,
                 ):
        '''Initialization of the controller.

        Args:
            initial_obs (ndarray): The initial observation of the quadrotor's state
                [x, x_dot, y, y_dot, z, z_dot, phi, theta, psi, p, q, r].
            initial_info (dict): The a priori information as a dictionary.
        '''
        # Save environment and conrol parameters.
        self.CTRL_FREQ = initial_info['ctrl_freq']
        self.CTRL_DT = 1.0 / self.CTRL_FREQ
        self.initial_obs = initial_obs

        # Create trajectory.
        self.full_trajectory = gen_traj(self.CTRL_FREQ, 20)
        self.lqr_gain = 0.05 * np.array([[4, 0.1]])

        self.prev_vel = 0
        self.prev_x = 0
        self.alpha = 0.3

        # Reset counters and buffers.
        self.reset()

    def cmdFirmware(self,
                    ep_time,
                    obs,
                    ):
        '''Pick command sent to the quadrotor through a Crazyswarm/Crazyradio-like interface.

        Args:
            ep_time (float): Episode's elapsed time, in seconds.
            obs (ndarray): The quadrotor's Vicon data [x, 0, y, 0, z, 0, phi, theta, psi, 0, 0, 0].
            reward (float, optional): The reward signal.
            done (bool, optional): Wether the episode has terminated.
            info (dict, optional): Current step information as a dictionary with keys
                'constraint_violation', 'current_target_gate_pos', etc.

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)
        '''

        iteration = int(ep_time * self.CTRL_FREQ)
        x_eq = 0  # -2.36
        y_eq = 0  # 0.762
        z_height = 0.5  # 1

        if iteration > 0:
            est_vel = (obs[0] - self.prev_x) / self.CTRL_DT
            self.prev_vel = (1 - self.alpha) * self.prev_vel + self.alpha * est_vel
        self.prev_x = obs[0]
        obs[1] = self.prev_vel

        if iteration == 0:
            print(f'Iter: {iteration} - Take off.')
            height = z_height
            duration = 2
            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration >= 2 * self.CTRL_FREQ and iteration < 3 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Re-centering at (x_eq, y_eq, z_height).')
            target_pos = np.array([x_eq, y_eq, z_height])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration >= 3 * self.CTRL_FREQ and iteration < 23 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Executing Trajectory.')
            step = iteration - 5 * self.CTRL_FREQ
            info = {'current_step': step}
            curr_obs = np.atleast_2d(np.array([obs[0] - x_eq, obs[1]])).T  # np.atleast_2d(np.array([obs[2]-y_eq, obs[3]])).T
            # new_act = -(-self.lqr_gain @ (curr_obs - self.full_trajectory[[step]].T))[0,0]
            new_act = self.ctrl.select_action(curr_obs, info)[0]
            new_act = np.clip(new_act, -0.25, 0.25)
            # new_act = -(self.full_trajectory[step, 0] - obs[0])
            if CERTIFY is True:
                certified_action, success = self.safety_filter.certify_action(curr_obs, new_act, info)
                if success:
                    self.corrections.append(certified_action - new_act)
                    new_act = certified_action
                else:
                    self.corrections.append(0.0)

            target_pos = np.array([new_act + obs[0], y_eq, z_height])  # np.array([x_eq, new_act + obs[2], z_height])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)

            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]

            self.recorded_obs.append(obs)
            self.actions.append(new_act)
        elif iteration >= 23 * self.CTRL_FREQ and iteration < 25 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Re-centering at (x_eq, y_eq, z_height).')
            target_pos = np.array([x_eq, y_eq, z_height])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration == 25 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Setpoint Stop.')
            command_type = Command(6)  # Notify setpoint stop.
            args = []
        elif iteration == 25 * self.CTRL_FREQ + 1:
            print(f'Iter: {iteration} - Landing.')
            height = 0.1
            duration = 2
            command_type = Command(3)  # Land.
            args = [height, duration]
        elif iteration == 27 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Terminating.')
            command_type = Command(-1)  # Terminate.
            args = []
            if save_data is True:
                if not CERTIFY:
                    folder = 'uncert'
                else:
                    folder = 'cert/' + sf_config['cost_function']
                    if sf_config['cost_function'] == 'precomputed_cost':
                        folder += '/m' + str(sf_config['mpsc_cost_horizon'])
                np.save(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/test{TEST}/{folder}/traj_goal.npy', self.full_trajectory)
                np.save(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/test{TEST}/{folder}/states.npy', np.array(self.recorded_obs))
                np.save(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/test{TEST}/{folder}/actions.npy', np.array(self.actions))
                np.save(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/test{TEST}/{folder}/corrections.npy', np.array(self.corrections))
        else:
            command_type = Command(0)  # None.
            args = []

        self.prev_obs = obs

        return command_type, args

    def setup_controllers(self):
        task_name = 'stab' if task_config['task'] == Task.STABILIZATION else 'track'

        task_config['gui'] = False
        env_func = partial(make,
                           task,
                           **task_config)

        # Setup controller.
        self.ctrl = make(algo,
                         env_func,
                         **algo_config)
        self.ctrl.gain = self.lqr_gain
        self.ctrl.model.U_EQ = 0

        self.ctrl.env.X_GOAL = self.full_trajectory
        self.ctrl.env.TASK = Task.TRAJ_TRACKING

        if CERTIFY is True:
            # Setup MPSC.
            self.safety_filter = make(sf,
                                      env_func,
                                      **sf_config)
            self.safety_filter.reset()
            self.safety_filter.load(path=f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/models/mpsc_parameters/{sf}_crazyflie_{task_name}.pkl')

            if sf_config['cost_function'] == Cost_Function.PRECOMPUTED_COST:
                self.safety_filter.cost_function.uncertified_controller = self.ctrl
                self.safety_filter.cost_function.output_dir = '/home/federico/GitHub/safe-control-gym/experiments/crazyflie'
                self.safety_filter.env.X_GOAL = self.full_trajectory

    def reset(self):
        '''Reset. '''
        self.recorded_obs = []
        self.actions = []
        self.corrections = []
        self.prev_obs = np.zeros((12, 1))
        self.setup_controllers()
