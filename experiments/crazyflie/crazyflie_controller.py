import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

from functools import partial
import pickle

import numpy as np

from experiments.crazyflie.crazyflie_utils import gen_traj
from safe_control_gym.controllers.firmware.firmware_wrapper import Command
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import mkdirs

save_data = True
TEST = 0
MODEL = 'mpsf_10'
CERTIFY = False

algo = 'ppo'
task = 'quadrotor'
sf = 'nl_mpsc'

algo_config = {
    'hidden_dim': 128,
    'activation': 'tanh',
    'norm_obs': False,
    'norm_reward': False,
    'clip_obs': 10,
    'clip_reward': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'use_clipped_value': False,
    'clip_param': 0.2,
    'target_kl': 0.01,
    'use_gae': True,
    'entropy_coef': 0.01,
    'opt_epochs': 20,
    'mini_batch_size': 256,
    'actor_lr': 0.001,
    'critic_lr': 0.001,
    'max_grad_norm': 0.5,
    'max_env_steps': 100000,
    'num_workers': 1,
    'rollout_batch_size': 1,
    'rollout_steps': 1000,
    'deque_size': 10,
    'eval_batch_size': 10,
    'log_interval': 5000,
    'save_interval': 0,
    'num_checkpoints': 0,
    'eval_interval': 5000,
    'eval_save_best': True,
    'tensorboard': False,
    'filter_train_actions': True,
    'penalize_sf_diff': True,
    'sf_penalty': 75,
    'use_safe_reset': True,
}

task_config = {
    'info_in_reset': True,
    'ctrl_freq': 25,
    'pyb_freq': 1000,
    'physics': 'pyb',
    'gui': False,
    'quad_type': 3,
    'normalized_rl_action_space': False,
    'episode_len_sec': 15,
    'inertial_prop':
    {'M': 0.03775,
     'Ixx': 1.4e-05,
     'Iyy': 1.4e-05,
     'Izz': 2.17e-05},
    'randomized_inertial_prop': False,
    'task': 'traj_tracking',
    'task_info':
    {
        'trajectory_type': 'figure8',
        'num_cycles': 1,
        'trajectory_plane': 'xz',
        'trajectory_position_offset': [0, 1],
        'trajectory_scale': 1,
        'proj_point': [0, 0, 0.5],
        'proj_normal': [0, 1, 1],
    },
    'cost': 'quadratic',
    'constraints':
    [
        {'constraint_form': 'bounded_constraint',
         'constrained_variable': 'state',
         'active_dims': [0,1,2,3,6,7],
         'upper_bounds': [0.95, 2, 0.95, 2, 0.25, 0.25],
         'lower_bounds': [-0.5, -2, -0.95, -2, -0.25, -0.25]},
        {'constraint_form': 'default_constraint',
         'constrained_variable': 'input',
         }
    ],
    'done_on_violation': False,
    'done_on_out_of_bound': True,
    'seed': 1337,
}

sf_config = {
    'r_lin': [2],
    'q_lin': [0.008, 1.85, 0.008, 1.85, 10, 10],
    'use_acados': True,
    'horizon': 10,
    'warmstart': True,
    'integration_algo': 'rk4',
    'use_terminal_set': False,
    'cost_function': 'precomputed_cost',
    'mpsc_cost_horizon': 10,
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

        # Create trajectory.
        self.full_trajectory = gen_traj(self.CTRL_FREQ, task_config['episode_len_sec'])

        self.prev_pos = np.squeeze(initial_obs)[[0,2,4,6,7]]
        self.prev_vel = np.squeeze(initial_obs)[[1,3,5,9,10]]
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

        Returns:
            Command: selected type of command (takeOff, cmdFullState, etc., see Enum-like class `Command`).
            List: arguments for the type of command (see comments in class `Command`)
        '''

        obs = np.array(obs)

        iteration = int(ep_time * self.CTRL_FREQ)
        z_height = 1

        self.low_pass_state(obs)

        if iteration == 0:
            print(f'Iter: {iteration} - Take off.')
            height = z_height
            duration = 2
            command_type = Command(2)  # Take-off.
            args = [height, duration]
        elif iteration >= 2 * self.CTRL_FREQ and iteration < 3 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Re-centering at (0, 0, z_height).')
            target_pos = np.array([0, 0, z_height])
            target_vel = np.zeros(3)
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration >= 3 * self.CTRL_FREQ and iteration < 18 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Executing Trajectory.')
            step = iteration - 3 * self.CTRL_FREQ
            info = {'current_step': step*20}

            curr_obs = obs[[0,1,2,3,6,7]].reshape((6, 1))

            new_act = np.squeeze(self.ctrl.select_action(curr_obs, info))

            if CERTIFY is True:
                certified_action, success = self.safety_filter.certify_action(curr_obs, new_act, info)
                if success:
                    self.corrections.append(certified_action - new_act)
                    # print(new_act, certified_action)
                    new_act = certified_action
                else:
                    self.corrections.append([0,0])

            command_type = Command(7)  # cmdVel
            args = [new_act[0]*180.0/np.pi, new_act[1]*180.0/np.pi, 0, 0]

            self.recorded_obs.append(obs)
            self.actions.append(new_act)
        elif iteration == 18 * self.CTRL_FREQ:
            self.return_path_pos = np.linspace(obs[[0, 2, 4]], [0, 0, z_height], 4 * self.CTRL_FREQ)
            self.return_path_vel = np.linspace(obs[[1, 3, 5]], [0, 0, 0], 4 * self.CTRL_FREQ)
            command_type = Command(1)  # cmdFullState.
            args = [self.return_path_pos[0], self.return_path_vel[0], np.zeros(3), 0.0, np.zeros(3)]
        elif iteration > 18 * self.CTRL_FREQ and iteration < 22 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Re-centering at (0, 0, z_height).')
            target_pos = self.return_path_pos[iteration - 18 * self.CTRL_FREQ]
            target_vel = self.return_path_pos[iteration - 18 * self.CTRL_FREQ]
            target_acc = np.zeros(3)
            target_yaw = 0.0
            target_rpy_rates = np.zeros(3)
            command_type = Command(1)  # cmdFullState.
            args = [target_pos, target_vel, target_acc, target_yaw, target_rpy_rates]
        elif iteration == 22 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Setpoint Stop.')
            command_type = Command(6)  # Notify setpoint stop.
            args = []
        elif iteration == 22 * self.CTRL_FREQ + 1:
            print(f'Iter: {iteration} - Landing.')
            height = 0.1
            duration = 2
            command_type = Command(3)  # Land.
            args = [height, duration]
        elif iteration == 24 * self.CTRL_FREQ:
            print(f'Iter: {iteration} - Terminating.')
            command_type = Command(-1)  # Terminate.
            args = []
            if save_data is True:
                if not CERTIFY:
                    folder = 'uncert'
                else:
                    folder = 'cert'

                pickle_data = {'states': np.array(self.recorded_obs), 'actions': np.array(self.actions), 'corrections': np.array(self.corrections), 'traj_goal': self.full_trajectory}
                mkdirs(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{MODEL}/{folder}')
                with open(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{MODEL}/{folder}/test{TEST}.pkl', 'wb') as file:
                    pickle.dump(pickle_data, file)
        else:
            command_type = Command(0)  # None.
            args = []

        return command_type, args


    def low_pass_state(self, obs):
        vel_mask = [1,3,5,9,10]
        pos_mask = [0,2,4,6,7]

        est_vel = (obs[pos_mask] - self.prev_pos) / self.CTRL_DT
        self.prev_vel = (1 - self.alpha) * self.prev_vel + self.alpha * est_vel
        self.prev_pos = obs[pos_mask]
        obs[vel_mask] = self.prev_vel


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
        if algo in ['ppo', 'sac', 'safe_explorer_ppo', 'cpo']:
            # Load state_dict from trained.
            self.ctrl.load(path=f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/models/rl_models/{algo}/{MODEL}/model_best.pt')

            # Remove temporary files and directories
            # self.shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)
            self.ctrl.X_GOAL = self.full_trajectory

        if CERTIFY is True:
            # Setup MPSC.
            self.safety_filter = make(sf,
                                      env_func,
                                      **sf_config)
            self.safety_filter.reset()
            self.safety_filter.load(path=f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/models/mpsc_parameters/{sf}_crazyflie_{task_name}.pkl')
            self.safety_filter.env.X_GOAL = self.full_trajectory

            if sf_config['cost_function'] == Cost_Function.PRECOMPUTED_COST:
                self.safety_filter.cost_function.uncertified_controller = self.ctrl
                self.safety_filter.cost_function.output_dir = '/home/federico/GitHub/safe-control-gym/experiments/crazyflie'

    def reset(self):
        '''Reset. '''
        self.recorded_obs = []
        self.actions = []
        self.corrections = []
        self.prev_pos = np.zeros((5,))
        self.prev_vel = np.zeros((5,))
        self.setup_controllers()
