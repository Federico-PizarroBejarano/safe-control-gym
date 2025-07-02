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


def plot_results(results):
    # Extract position data
    positions = np.array([obs.pos.squeeze() for obs in results])
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


def control(obs, t, i_error, dt, start_pos):
    pos, vel, quat = obs.pos, obs.vel, obs.quat
    des_pos = start_pos + np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd, i_error = jit_state2attitude(
        pos, vel, quat, des_pos, np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), i_error, dt
    )
    return cmd, i_error


def run(num_drones=1, duration=5.0, fps=60, safety_filter=None):
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

    # Run the simulation.
    i_error = np.zeros((1, 1, 3))
    all_obs = []
    start_pos = sim.data.states.pos
    for i in range(int(duration * sim.control_freq)):
        # Get the current state.
        obs = sim.data.states
        all_obs.append(obs)
        rpys = []
        for drone_idx in range(num_drones):
            rpy = RotLib.from_quat(obs.quat[0, drone_idx, :].flatten()).as_euler('xyz')
            rpys.append(rpy)
        rpys = np.array(rpys).reshape((1, num_drones, 3))
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpys, obs.ang_vel], axis=-1).squeeze()
        stacked_obs = stacked_obs[:, np.array([0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11])]
        stacked_obs = stacked_obs.flatten()

        # Compute the control command.
        cmd, i_error = control(obs, i * dt, i_error, dt, start_pos)
        cmd, _ = safety_filter.certify_action(stacked_obs, cmd.flatten())

        # Apply the control command.
        sim.attitude_control(cmd.reshape(1, num_drones, -1))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()

    plot_results(all_obs)


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

    run(num_drones=config.num_drones, duration=5.0, fps=60, safety_filter=safety_filter)


if __name__ == '__main__':
    main()
