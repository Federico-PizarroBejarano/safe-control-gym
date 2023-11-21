'''Running MPSC using the crazyflie firmware. '''

import sys
import time
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from experiments.crazyflie.crazyflie_utils import gen_traj
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


A = np.array([[0.9987, 0.02872],
              [0.006117, 0.8535]])
B = np.array([[0.02309, 0.2854]]).T


def run(gui=False, plot=True, training=False, certify=False, traj='sine', curr_path='.'):
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
    env_func_500 = partial(make,
                           config.task,
                           **config.task_config)

    states = []
    actions_uncert = []
    actions_cert = []

    errors = []

    # Create environment.
    firmware_wrapper = make('firmware', env_func_500, FIRMWARE_FREQ, CTRL_FREQ)
    obs, info = firmware_wrapper.reset()
    env = firmware_wrapper.env

    # Create trajectory.
    full_trajectory = gen_traj(CTRL_FREQ, env.EPISODE_LEN_SEC)
    lqr_gain = 0.05 * np.array([[4, 0.1]])

    # Setup controller.
    ctrl = make(config.algo,
                env_func,
                **config.algo_config)
    ctrl.gain = lqr_gain
    ctrl.model.U_EQ = 0

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
            1 / 0
        else:
            safety_filter.load(path=f'{curr_path}/models/mpsc_parameters/{config.safety_filter}_crazyflie_{task}.pkl')

        if config.sf_config.cost_function == Cost_Function.PRECOMPUTED_COST:
            safety_filter.cost_function.uncertified_controller = ctrl
            safety_filter.cost_function.output_dir = curr_path
            safety_filter.env.X_GOAL = full_trajectory

    ep_start = time.time()
    states.append(env.state)
    action = env.U_GOAL
    successes = 0
    estimated_vel = []
    bad_estimated_vel = []
    prev_vel = 0
    prev_x = 0
    alpha = 0.3
    for i in range(CTRL_FREQ * env.EPISODE_LEN_SEC):
        curr_obs = np.atleast_2d(np.array([obs[0], obs[1]])).T
        info['current_step'] = i
        new_act = ctrl.select_action(curr_obs, info)[0]
        new_act = np.clip(new_act, -0.25, 0.25)
        actions_uncert.append(new_act)
        if certify is True:
            certified_action, success = safety_filter.certify_action(curr_obs, new_act, info)
            if success:
                successes += 1
                new_act = certified_action
        actions_cert.append(new_act)
        next_state = A @ curr_obs + B * new_act
        pos = [(new_act + curr_obs[0])[0], 0, 1]
        vel = [0, 0, 0]
        acc = [0, 0, 0]
        yaw = 0
        rpy_rate = [0, 0, 0]
        args = [pos, vel, acc, yaw, rpy_rate]

        curr_time = i * CTRL_DT
        firmware_wrapper.sendFullStateCmd(*args, curr_time)

        # Step the environment.
        obs, _, _, info, action = firmware_wrapper.step(curr_time, action)
        x_obs = obs[0] + np.random.normal(0.0, 0.001)
        est_vel = (x_obs - prev_x) / CTRL_DT
        bad_estimated_vel.append(est_vel)
        prev_vel = (1 - alpha) * prev_vel + alpha * est_vel
        prev_x = x_obs
        estimated_vel.append(prev_vel)
        obs[0] = x_obs
        obs[1] = prev_vel
        errors.append(np.squeeze(np.array([obs[0], obs[1]])) - np.squeeze(next_state))

        states.append(obs)
        if obs[4] < 0.05:
            print('CRASHED!!!')
            break

        # Synchronize the GUI.
        if config.task_config.gui:
            sync(i, ep_start, CTRL_DT)

    states = np.array(states)
    actions_uncert = np.array(actions_uncert)
    print('Number of Max Inputs: ', np.sum(np.abs(actions_uncert) == 0.25))
    actions_cert = np.array(actions_cert)
    errors = np.array(errors)
    corrections = actions_cert - actions_uncert

    # Close the environment
    env.close()
    print('Elapsed Time: ', time.time() - ep_start)
    print('Model Errors: ', np.linalg.norm(errors))
    print(f'Feasible steps: {successes}/{CTRL_FREQ*env.EPISODE_LEN_SEC}')
    print('NUM ERRORS POS: ', np.sum(np.abs(states[:, 0]) >= 0.75))
    print('NUM ERRORS VEL: ', np.sum(np.abs(states[:, 1]) >= 0.5))
    print('Rate of change (inputs): ', np.linalg.norm(get_discrete_derivative(np.atleast_2d(actions_cert).T, CTRL_FREQ)))
    print('Max Correction: ', np.max(np.abs(corrections)))
    print('Magnitude of Corrections: ', np.linalg.norm(corrections))

    if certify is False:
        np.save('./models/results/states_uncert.npy', states)
        np.save('./models/results/actions_uncert.npy', actions_uncert)
        np.save('./models/results/errors_uncert.npy', errors)
    else:
        np.save('./models/results/states_cert.npy', states)
        np.save('./models/results/actions_uncert.npy', actions_uncert)
        np.save('./models/results/actions_cert.npy', actions_cert)
        np.save('./models/results/errors_cert.npy', errors)

    if plot:
        plt.plot(states[:, 0], label='x')
        plt.plot(states[:, 2], label='y')
        plt.plot(states[:, 4], label='z')
        plt.legend()
        plt.show()

        plt.plot(states[:, 0], label='traj')
        plt.plot(full_trajectory[:, 0], label='ref')
        plt.legend()
        plt.show()

        plt.plot(states[:, 1], label='vel x')
        plt.plot(full_trajectory[:, 1], label='ref vel')
        plt.plot(estimated_vel, label='est vel')
        plt.plot(bad_estimated_vel, label='bad est vel')
        plt.legend()
        plt.show()

    print('Experiment Complete.')
    savemat(f'{curr_path}/models/results/matlab_data.mat', {'states': states, 'actions': actions_cert})


if __name__ == '__main__':
    run()
