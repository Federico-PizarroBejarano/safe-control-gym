'''Running MPSC using the crazyflie firmware. '''

import sys
import shutil
import time
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from experiments.crazyflie.crazyflie_utils import gen_traj, gen_input_traj
from safe_control_gym.envs.benchmark_env import Task
from safe_control_gym.safety_filters.mpsc.mpsc_utils import Cost_Function, get_discrete_derivative
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make
from safe_control_gym.utils.utils import sync

try:
    import pycffirmware
except ImportError:
    FIRMWARE_INSTALLED = False
else:
    FIRMWARE_INSTALLED = True
finally:
    print('Module \'cffirmware\' available:', FIRMWARE_INSTALLED)


A = np.array([[ 1,           0.03659,   7.598e-07, -0.006083],
              [-5.858e-06,   0.7886,    5.132e-06,  0.03174],
              [ 3.259e-06,   0.0009138, 1,          0.03899],
              [-1.735e-05,  -0.006111, -9.836e-06,  0.7786]])

B = np.array([[ 0.003886,  0.01169],
              [ 0.4229,   -0.06055],
              [-0.001915, -0.0006503],
              [ 0.01223,   0.4419]])


def run(gui=False, plot=True, training=False, certify=True, curr_path='.'):
    '''The main function creating, running, and closing an environment over N episodes. '''

    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()
    task = 'stab' if config.task_config.task == Task.STABILIZATION else 'track'

    CTRL_FREQ = config.task_config['ctrl_freq']
    CTRL_DT = 1 / CTRL_FREQ

    env_func = partial(make,
                       config.task,
                       **config.task_config)

    FIRMWARE_FREQ = 500

    config.task_config.gui = gui
    config.task_config['ctrl_freq'] = FIRMWARE_FREQ
    config.algo_config['training'] = False

    env_func_500 = partial(make,
                           config.task,
                           **config.task_config)

    states = []
    actions_uncert = []
    actions_cert = []

    # Create environment.
    firmware_wrapper = make('firmware', env_func_500, FIRMWARE_FREQ, CTRL_FREQ)
    obs, info = firmware_wrapper.reset()
    env = firmware_wrapper.env

    # Create trajectory.
    full_trajectory = gen_traj(CTRL_FREQ, env.EPISODE_LEN_SEC)
    full_trajectory = np.hstack((full_trajectory, full_trajectory))

    lqr_gain = 0.05 * np.array([[4,  0.1,  0,  0],
                                [0,  0,    4,  0.1]])

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config)

    if config.algo in ['ppo', 'sac', 'safe_explorer_ppo', 'cpo']:
        # Load state_dict from trained.
        ctrl.load(f'{curr_path}/{config.output_dir}/model_latest.pt')

        # Remove temporary files and directories
        shutil.rmtree(f'{curr_path}/temp', ignore_errors=True)
        ctrl.X_GOAL = full_trajectory
    else:
        ctrl.gain = lqr_gain
        ctrl.model.U_EQ = np.array([[0, 0]]).T

        ctrl.env.X_GOAL = full_trajectory
        ctrl.env.TASK = Task.TRAJ_TRACKING

    if certify is True:
        # Setup MPSC.
        safety_filter = make(config.safety_filter,
                             env_func,
                             **config.sf_config)
        safety_filter.reset()
        if training is True:
            safety_filter.learn(env=env)
            safety_filter.save(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_crazyflie_{task}.pkl')
            1/0
        else:
            safety_filter.load(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_crazyflie_{task}.pkl')
            safety_filter.env.X_GOAL = full_trajectory

        if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
            safety_filter.cost_function.uncertified_controller = ctrl
            safety_filter.cost_function.output_dir = curr_path
            safety_filter.env.X_GOAL = full_trajectory

    ep_start = time.time()
    states.append(env.state)
    action = env.U_GOAL
    successes = 0

    for i in range(CTRL_FREQ * env.EPISODE_LEN_SEC):
        curr_obs = np.atleast_2d(obs[0:4]).T
        curr_obs = curr_obs.reshape((4, 1))
        new_act = np.squeeze(ctrl.select_action(curr_obs, info))
        new_act = np.clip(new_act, np.array([-0.25, -0.25]), np.array([0.25, 0.25]))
        actions_uncert.append(new_act)
        if certify is True:
            certified_action, success = safety_filter.certify_action(curr_obs, new_act, info)
            if success:
                successes += 1
                new_act = certified_action
        actions_cert.append(new_act)
        pos = [(new_act[0] + curr_obs[0])[0], (new_act[1] + curr_obs[2])[0], 1]
        vel = [0, 0, 0]
        acc = [0, 0, 0]
        yaw = 0
        rpy_rate = [0, 0, 0]
        args = [pos, vel, acc, yaw, rpy_rate]

        curr_time = i * CTRL_DT
        firmware_wrapper.sendFullStateCmd(*args, curr_time)

        # Step the environment.
        obs, _, _, info, action = firmware_wrapper.step(curr_time, action)

        states.append(obs)
        if obs[4] < 0.05:
            print('CRASHED!!!')
            break

        # Synchronize the GUI.
        if config.task_config.gui:
            sync(i, ep_start, CTRL_DT)

    states = np.array(states)
    actions_uncert = np.array(actions_uncert)
    print(f'Number of Max Inputs: {np.sum(np.abs(actions_uncert) == 0.25)}/{2*len(actions_uncert)}')
    actions_cert = np.array(actions_cert)
    corrections = np.squeeze(actions_cert) - np.squeeze(actions_uncert)

    # Close the environment
    env.close()
    print('Elapsed Time: ', time.time() - ep_start)
    print('NUM VIOLATIONS POS: ', np.sum(np.abs(states[:, 0]) >= 0.75))
    print('NUM VIOLATIONS VEL: ', np.sum(np.abs(states[:, 1]) >= 0.5))
    print('Rate of change (inputs): ', np.linalg.norm(get_discrete_derivative(np.atleast_2d(actions_cert).T, CTRL_FREQ)))
    if certify:
        print(f'Feasible steps: {float(successes)}/{CTRL_FREQ*env.EPISODE_LEN_SEC}')
        print('Max Correction: ', np.max(np.abs(corrections)))
        print('Magnitude of Corrections: ', np.linalg.norm(corrections))

    if plot:
        plt.plot(states[:, 0], label='x')
        plt.plot(states[:, 2], label='y')
        plt.plot(states[:, 4], label='z')
        plt.legend()
        plt.show()

        plt.plot(states[:, 1], label='x vel')
        plt.plot(states[:, 3], label='y vel')
        plt.plot(states[:, 5], label='z vel')
        plt.legend()
        plt.show()

        plt.plot(states[:, 0], label='x traj')
        plt.plot(states[:, 2], label='y traj')
        plt.plot(full_trajectory[:, 0], label='ref')
        plt.legend()
        plt.show()

    print('Experiment Complete.')


def identify_system(curr_path='.'):
    '''The main function creating, running, and closing an environment over N episodes. '''
    # Define arguments.
    fac = ConfigFactory()
    config = fac.merge()

    CTRL_FREQ = config.task_config['ctrl_freq']
    CTRL_DT = 1 / CTRL_FREQ

    FIRMWARE_FREQ = 500
    config.task_config['ctrl_freq'] = FIRMWARE_FREQ
    env_func_500 = partial(make,
                           config.task,
                           **config.task_config)

    states = []
    actions = []

    # Create environment.
    firmware_wrapper = make('firmware', env_func_500, FIRMWARE_FREQ, CTRL_FREQ)
    obs, _ = firmware_wrapper.reset()
    env = firmware_wrapper.env

    # Create trajectory.
    input_traj = gen_input_traj(CTRL_FREQ, env.EPISODE_LEN_SEC, num_channels=2)

    states.append(env.state)
    action = env.U_GOAL

    errors = []

    for i in range(10000):
        curr_obs = np.atleast_2d(obs[0:4]).T
        new_act = np.atleast_2d(input_traj[:, i]).T
        actions.append(new_act)

        pos = [(new_act[0] + curr_obs[0])[0], (new_act[1] + curr_obs[2])[0], 1]
        args = [pos, [0, 0, 0], [0, 0, 0], 0, [0, 0, 0]]

        curr_time = i * CTRL_DT
        firmware_wrapper.sendFullStateCmd(*args, curr_time)

        # Step the environment.
        obs, _, _, _, action = firmware_wrapper.step(curr_time, action)
        states.append(obs.copy())

        pred_next_state = A @ curr_obs + B @ new_act
        errors.append(np.squeeze(obs[0:4] - np.squeeze(pred_next_state)))

        if obs[4] < 0.05:
            print('CRASHED!!!')
            break

    states = np.array(states)
    actions = np.array(actions)
    errors = np.array(errors)

    normed_w = np.linalg.norm(errors, axis=1)
    print('MAX ERROR:', np.max(normed_w))
    print('MEAN ERROR:', np.mean(normed_w))
    print('MAX ERROR PER DIM:', np.max(errors, axis=0))
    print('TOTAL ERRORS BY CHANNEL:', np.sum(np.abs(errors), axis=0))

    # Close the environment
    env.close()

    print('Experiment Complete.')
    savemat(f'{curr_path}/models/traj_data/matlab_data.mat', {'states': states, 'actions': actions})
    np.save('./models/traj_data/errors.npy', errors)

    plt.plot(states[:, 0], label='x')
    plt.plot(states[:, 2], label='y')
    plt.plot(states[:, 4], label='z')
    plt.legend()
    plt.show()

    plt.plot(states[:, 1], label='x')
    plt.plot(states[:, 3], label='y')
    plt.plot(states[:, 5], label='z')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    run()
    # identify_system()
