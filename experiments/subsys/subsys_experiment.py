import time
from functools import partial

import jax
import matplotlib.pyplot as plt
import numpy as np
from crazyflow.control import Control
from crazyflow.control.control import state2attitude
from crazyflow.sim import Sim
from scipy.spatial.transform import Rotation as RotLib

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

jit_state2attitude = jax.jit(state2attitude)


def plot_results(num_drones, results):
    # Extract position data
    positions = np.array([obs.pos.squeeze() for obs in results]).reshape((-1, num_drones, 3))
    x = positions[:, :, 0]
    y = positions[:, :, 1]

    # Get constraint bounds from safety filter config
    x_bounds = [-2.5, 0.5]
    y_bounds = [-1.5, 1.5]

    # Plot trajectory and constraints
    plt.figure(figsize=(8, 8))
    for drone_idx in range(positions.shape[1]):
        plt.plot(x[:, drone_idx], y[:, drone_idx], label=f'Trajectory_{drone_idx}')

    if x_bounds is not None and y_bounds is not None:
        plt.axhline(y=y_bounds[0], color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=y_bounds[1], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=x_bounds[0], color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=x_bounds[1], color='r', linestyle='--', alpha=0.5)

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


def generate_X_goal(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    X_goal = np.zeros((num_iters, num_drones, 12))
    for i in range(num_iters):
        mag = 1.001**i
        next_pos = start_pos + np.array([mag * np.cos(i * dt) - 1, mag * np.sin(i * dt), 1 - np.cos(i * dt)])
        goal_state = np.zeros((num_drones, 12))
        goal_state[:, [0, 2, 4]] = next_pos
        X_goal[i, :] = goal_state
    return X_goal


def control(obs, i_error, des_pos, dt):
    des_pos = des_pos[:, 0:5:2].reshape((1, -1, 3))
    pos, vel, quat = obs.pos, obs.vel, obs.quat
    cmd, i_error = jit_state2attitude(
        pos, vel, quat, des_pos, np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), i_error, dt,
    )
    return cmd.flatten(), i_error


def run(plot=False, num_drones=1, duration=5.0, fps=60, safety_filter=None):
    # Create the simulation environment.
    sim = Sim(
        n_drones=num_drones,
        control=Control.attitude,
        attitude_freq=25,
        integrator='rk4',
        physics='sys_id',
    )
    sim.reset()
    dt = 1 / sim.control_freq

    new_pos = sim.data.states.pos.at[:, :, 2].set(0.1)
    new_pos = new_pos.at[:, :, 0:2].multiply(0.95)
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=new_pos)
    )

    X_goal = generate_X_goal(sim.data.states.pos[0, :, :], int(duration * sim.control_freq), dt)

    # Run the simulation.
    i_error = np.zeros((1, 1, 3))
    all_obs = []
    all_corrections = []
    start_time = time.time()
    for i in range(int(duration * sim.control_freq)):
        # Get the current state.
        obs = sim.data.states
        all_obs.append(obs)
        rpys = []
        for drone_idx in range(num_drones):
            rpy = RotLib.from_quat(obs.quat[0, drone_idx, :].flatten()).as_euler('xyz')
            rpys.append(rpy)
        rpys = np.array(rpys).reshape((1, num_drones, 3))
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpys, obs.ang_vel], axis=-1)[0, :, :]
        stacked_obs = stacked_obs[:, np.array([0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11])]
        stacked_obs = stacked_obs.flatten()

        # Compute the control command.
        uncert_cmd, i_error = control(obs, i_error, X_goal[i, :, :], dt)
        cert_cmd, _ = safety_filter.certify_action(stacked_obs, uncert_cmd)
        all_corrections.append(np.linalg.norm(uncert_cmd - cert_cmd))

        # Apply the control command.
        sim.attitude_control(cert_cmd.reshape(1, num_drones, -1))
        sim.step(sim.freq // sim.control_freq)
        if i == 0:
            start_time = time.time()
        if plot and ((i * fps) % sim.control_freq) < fps:
            sim.render()
    print(f'Time taken: {time.time() - start_time} seconds')
    sim.close()

    print('Mean Correction:', np.round(np.mean(all_corrections), 3))
    print('Max Correction:', np.round(np.max(all_corrections), 3))

    plot_results(num_drones, all_obs)


def main():
    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Setup MPSC.
    safety_filter = make(config.safety_filter,
                         env_func,
                         num_drones=config.num_drones,
                         **config.sf_config)
    safety_filter.reset()

    run(
        plot=False,
        num_drones=config.num_drones,
        duration=15.0,
        fps=60,
        safety_filter=safety_filter,
    )


if __name__ == '__main__':
    main()
