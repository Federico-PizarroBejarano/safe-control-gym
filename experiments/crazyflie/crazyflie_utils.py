import sys
sys.path.insert(0, '/home/federico/GitHub/safe-control-gym')

import matplotlib.pyplot as plt
import numpy as np

from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative


def calc_error(CTRL_FREQ, EPISODE_LEN_SEC, TEST=0, CERTIFIED=False, COST_FUNCTION='one_step', M=2):
    if not CERTIFIED:
        folder = 'uncert'
    else:
        folder = 'cert/' + COST_FUNCTION + '_cost'
        if COST_FUNCTION == 'precomputed':
            folder += '/m' + str(M)
    states = np.load(f'./all_trajs/test{TEST}/{folder}/states.npy')
    actions = np.load(f'./all_trajs/test{TEST}/{folder}/actions.npy')
    corrections = np.load(f'./all_trajs/test{TEST}/{folder}/corrections.npy')

    A = np.array([[0.9987, 0.02872],
                  [0.006117, 0.8535]])
    B = np.array([[0.02309, 0.2854]]).T

    errors = []
    for i in range(len(states) - 1):
        curr_obs = np.array([states[i, 0], states[i, 1]]).T
        next_obs = np.array([states[i + 1, 0], states[i + 1, 1]]).T
        next_est = np.squeeze(A @ curr_obs) + np.squeeze(B * actions[i])
        errors.append(np.squeeze(next_obs) - np.squeeze(next_est))

    print('Model Errors: ', np.linalg.norm(errors))
    print('NUM ERRORS POS: ', np.sum(np.abs(states[:, 0]) >= 0.75))
    print('NUM ERRORS VEL: ', np.sum(np.abs(states[:, 1]) >= 0.5))
    print('Rate of change (inputs): ', np.linalg.norm(get_discrete_derivative(actions.reshape(-1, 1), CTRL_FREQ)))

    if CERTIFIED:
        print('Max Correction: ', np.max(np.abs(corrections)))
        print('Magnitude of Corrections: ', np.linalg.norm(corrections))


def plot_traj(CTRL_FREQ, TEST=0, CERTIFIED=False, COST_FUNCTION='one_step', M=2):
    if not CERTIFIED:
        folder = 'uncert'
    else:
        folder = 'cert/' + COST_FUNCTION + '_cost'
        if COST_FUNCTION == 'precomputed':
            folder += '/m' + str(M)

    states = np.load(f'./all_trajs/test{TEST}/{folder}/states.npy')
    goal_traj = np.load(f'./all_trajs/test{TEST}/{folder}/traj_goal.npy')
    goal_traj *= -1

    estimated_vel = []
    bad_estimated_vel = []
    prev_vel = 0
    alpha = 0.3
    for x in range(1, len(states[:, 0])):
        est_vel = (states[x, 0] - states[x - 1, 0]) * CTRL_FREQ
        bad_estimated_vel.append(est_vel)
        prev_vel = (1 - alpha) * prev_vel + alpha * est_vel
        estimated_vel.append(prev_vel)

    plt.plot(states[:, 0], label='x')
    plt.plot(states[:, 2], label='y')
    plt.plot(states[:, 4], label='z')
    plt.legend()
    plt.show()

    plt.plot(states[:, 0], label='traj')
    plt.plot(goal_traj[:, 0], label='ref')
    plt.legend()
    plt.show()

    plt.plot(states[:, 1], label='vel x')
    plt.plot(goal_traj[:, 1], label='ref vel')
    plt.plot(estimated_vel, label='est vel')
    plt.plot(bad_estimated_vel, label='bad est vel')
    plt.legend()
    plt.show()


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

    linspace = np.linspace(0, 4 * np.pi, CTRL_FREQ * EPISODE_LEN_SEC)
    trajectory = 1.5 * np.sin(linspace)
    x = np.array(range(CTRL_FREQ * EPISODE_LEN_SEC))
    ramp_up = 1 / (1 + np.exp(-0.03 * (x - 100)))
    trajectory = ramp_up * trajectory

    traj_vel = (trajectory[2:] - trajectory[:-2]) / (CTRL_DT * 2)
    traj_vel = np.insert(traj_vel, 0, (trajectory[1] - trajectory[0]) / CTRL_DT)
    traj_vel = np.append(traj_vel, (trajectory[-1] - trajectory[-2]) / CTRL_DT)
    full_trajectory = np.hstack((np.atleast_2d(trajectory).T, np.atleast_2d(traj_vel).T))

    if plot is True:
        plt.plot(full_trajectory[:, 0], label='ref')
        plt.legend()
        plt.show()

        plt.plot(full_trajectory[:, 1], label='ref vel')
        plt.legend()
        plt.show()

    return full_trajectory


def get_max_chatter(CERTIFIED, COST_FUNCTION, M):
    chatter_widths = []
    for TEST in range(0, 5):
        if not CERTIFIED:
            folder = 'uncert'
        else:
            folder = 'cert/' + COST_FUNCTION + '_cost'
            if COST_FUNCTION == 'precomputed':
                folder += '/m' + str(M)

        states = np.load(f'./all_trajs/test{TEST}/{folder}/states.npy')
        min_x = np.min(states[:, 0])
        min_index = np.argmin(states[:, 0])
        max_x = np.max(states[min_index:min_index + 20, 0])
        chatter_widths.append(max_x - min_x)

    chatter_widths = np.array(chatter_widths)
    print(f'Max Chatter: {chatter_widths.mean()} w STD: {chatter_widths.std()}')


if __name__ == '__main__':
    TEST = 0
    CERTIFIED = True
    COST_FUNCTION = 'precomputed'
    M = 10

    # gen_traj(CTRL_FREQ=25, EPISODE_LEN_SEC=20, plot=True)
    gen_input_traj(CTRL_FREQ=25, EPISODE_LEN_SEC=20, plot=True)
    # plot_traj(CTRL_FREQ=25, TEST=TEST, CERTIFIED=CERTIFIED, COST_FUNCTION=COST_FUNCTION, M=M)
    # get_max_chatter(CERTIFIED=CERTIFIED, COST_FUNCTION=COST_FUNCTION, M=M)
    # calc_error(CTRL_FREQ=25, EPISODE_LEN_SEC=20, TEST=TEST, CERTIFIED=CERTIFIED, COST_FUNCTION=COST_FUNCTION, M=M)
