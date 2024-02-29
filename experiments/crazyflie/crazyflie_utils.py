import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

import pickle

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative


def calc_error(CTRL_FREQ, TEST=0, CERTIFIED=False, MODEL='none'):
    if not CERTIFIED:
        folder = 'uncert'
    else:
        folder = 'cert'

    with open(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{MODEL}/{folder}/test{TEST}.pkl', 'rb') as file:
        pickle_data = pickle.load(file)
    states = pickle_data['states']
    actions = pickle_data['actions']
    ref_traj = pickle_data['traj_goal']
    corrections = pickle_data['corrections']

    A = np.array([[0.9885, 0.0419, -0.0005, 0.0, -0.0032, 0.0354],
            [-0.0862, 1.0142, -0.0037, 0.0004, -0.0236, 0.2658],
            [-0.0015, 0.0002, 0.9979, 0.0399, -0.0485, -0.0031],
            [-0.0112, 0.0012, -0.0157, 0.9994, -0.3639, -0.0232],
            [-0.0109, 0.0018, -0.0111, 0.0017, 0.8245, -0.0036],
            [0.0399, -0.0072, -0.0013, 0.0008, -0.0156, 0.8651]])
    B = np.array([[0.0011, -0.0015],
            [0.0082, -0.0111],
            [0.0083, -0.002],
            [0.0624, -0.0147],
            [0.1944, -0.0307],
            [0.0344, 0.1793]])

    errors = []
    for i in range(len(states) - 1):
        curr_obs = np.array([states[i, [0,1,2,3,6,7]]]).T
        next_obs = np.array([states[i + 1, [0,1,2,3,6,7]]]).T
        next_est = np.squeeze(A @ curr_obs) + np.squeeze(B @ actions[i, :])
        errors.append(np.squeeze(next_obs) - np.squeeze(next_est))

    error = states[:, [0,2]] - ref_traj[:, [0,2]]
    dist = np.sum(2 * error * error, axis=1)
    reward = np.sum(np.exp(-dist))

    print('Model Errors: ', np.linalg.norm(errors))
    print('NUM ERRORS POS: ', np.sum(np.abs(states[:, [0,2]]) >= 0.95))
    print('NUM ERRORS VEL: ', np.sum(np.abs(states[:, [1,3]]) >= 2))
    print('NUM ERRORS ANGLE: ', np.sum(np.abs(states[:, [6,7]]) >= 0.25))
    print('Rate of change (inputs): ', np.linalg.norm(get_discrete_derivative(actions.reshape(-1, 1), CTRL_FREQ)))
    print('Reward: ', reward)

    if CERTIFIED:
        print('Max Correction: ', np.max(np.abs(corrections)))
        print('Magnitude of Corrections: ', np.linalg.norm(corrections))


def plot_traj(CTRL_FREQ, TEST=0, CERTIFIED=False, MODEL='none'):
    if not CERTIFIED:
        folder = 'uncert'
    else:
        folder = 'cert'

    with open(f'/home/federico/GitHub/safe-control-gym/experiments/crazyflie/all_trajs/{MODEL}/{folder}/test{TEST}.pkl', 'rb') as file:
        pickle_data = pickle.load(file)
    states = pickle_data['states']
    goal_traj = pickle_data['traj_goal']

    estimated_vel = []
    bad_estimated_vel = []
    prev_vel = 0
    alpha = 0.3
    for x in range(1, len(states[:, 0])):
        est_vel = (states[x, 0] - states[x - 1, 0]) * CTRL_FREQ
        bad_estimated_vel.append(est_vel)
        prev_vel = (1 - alpha) * prev_vel + alpha * est_vel
        estimated_vel.append(prev_vel)

    # plt.plot(states[:, 0], label='x')
    # plt.plot(states[:, 2], label='y')
    # plt.plot(states[:, 4], label='z')
    # plt.legend()
    # plt.show()

    # plt.plot(states[:, 0], label='x traj')
    # plt.plot(goal_traj[:, 0], label='x ref')
    # plt.plot(states[:, 2], label='y traj')
    # plt.plot(goal_traj[:, 2], label='y ref')
    # plt.legend()
    # plt.show()

    plt.plot(states[:, 0], states[:, 2], label='x-y')
    plt.plot(goal_traj[:, 0], goal_traj[:, 2], label='ref')
    plt.legend()
    plt.show()

    # plt.plot(actions[:, 0], label='roll cmd')
    # plt.plot(actions[:, 1], label='pitch cmd')
    # plt.legend()
    # plt.show()

    # plt.plot(states[:, 1], label='x vel')
    # plt.plot(states[:, 3], label='y vel')
    # plt.plot(states[:, 5], label='z vel')
    # plt.legend()
    # plt.show()

    # plt.plot(states[:, 6], label='roll')
    # plt.plot(states[:, 7], label='pitch')
    # plt.legend()
    # plt.show()

    # plt.plot(states[:, 1], label='vel x')
    # plt.plot(goal_traj[:, 1], label='ref vel')
    # plt.plot(estimated_vel, label='est vel')
    # plt.plot(bad_estimated_vel, label='bad est vel')
    # plt.legend()
    # plt.show()


def gen_input_traj(CTRL_FREQ, EPISODE_LEN_SEC, num_channels=1, plot=False):
    num_freqs = 20

    input_traj = []
    for _ in range(num_channels):
        freqs = np.power(np.random.rand(num_freqs+1)*2, 2)
        freqs = np.linspace(freqs[:-1], freqs[1:], CTRL_FREQ * EPISODE_LEN_SEC, axis=1).flatten()
        x = np.linspace(0, 12*np.pi, num_freqs * CTRL_FREQ * EPISODE_LEN_SEC)
        traj = np.sin(np.multiply(x, freqs))
        input_traj.append(traj*0.25)

    if plot:
        plt.plot(traj)
        plt.show()

    return np.array(input_traj)


def gen_traj(CTRL_FREQ, EPISODE_LEN_SEC, plot=False):
    CTRL_DT = 1.0 / CTRL_FREQ

    x = np.sin(np.linspace(0, 8*np.pi, CTRL_FREQ*EPISODE_LEN_SEC))
    y = np.sin(np.linspace(2*np.pi, 6*np.pi, CTRL_FREQ*EPISODE_LEN_SEC))

    ramp_up = np.concatenate((np.linspace(0, 1, CTRL_FREQ*2), np.ones((EPISODE_LEN_SEC-2)*CTRL_FREQ)))
    x = x * ramp_up
    y = y * ramp_up

    x_vel = (x[2:] - x[:-2]) / (CTRL_DT * 2)
    x_vel = np.insert(x_vel, 0, (x[1] - x[0]) / CTRL_DT)
    x_vel = np.append(x_vel, (x[-1] - x[-2]) / CTRL_DT)

    y_vel = (y[2:] - y[:-2]) / (CTRL_DT * 2)
    y_vel = np.insert(y_vel, 0, (y[1] - y[0]) / CTRL_DT)
    y_vel = np.append(y_vel, (y[-1] - y[-2]) / CTRL_DT)

    full_trajectory = np.vstack((x, x_vel, y, y_vel)).T

    print(f'Max x: {np.max(x)}, Max x_vel: {np.max(x_vel)}, Max y: {np.max(y)}, Max y_vel: {np.max(y_vel)}')

    if plot is True:
        plt.plot(full_trajectory[:, 0], label='x')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 1], label='x_vel')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 2],label='y')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 3], label='y_vel')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 0], full_trajectory[:, 2], label='ref')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 1], full_trajectory[:, 3], label='ref vel')
        plt.legend()
        plt.show()

    return full_trajectory


if __name__ == '__main__':
    CERTIFIED = True
    suffix = '_dm'

    # gen_traj(CTRL_FREQ=25, EPISODE_LEN_SEC=15, plot=True)
    # gen_input_traj(CTRL_FREQ=25, EPISODE_LEN_SEC=20, plot=True)
    # get_max_chatter(CERTIFIED=CERTIFIED, COST_FUNCTION=COST_FUNCTION, M=M)

    for model in ['mpsf_0.1', 'mpsf_1', 'mpsf_10', 'none_cpen', 'none']:
        MODEL = model + suffix
        for test in range(5):
            print(f'MODEL: {MODEL}, TEST: {test}')
            plot_traj(CTRL_FREQ=25, TEST=test, CERTIFIED=CERTIFIED, MODEL=MODEL)
            calc_error(CTRL_FREQ=25, TEST=test, CERTIFIED=CERTIFIED, MODEL=MODEL)
