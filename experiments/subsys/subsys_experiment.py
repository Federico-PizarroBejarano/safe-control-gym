from functools import partial

import jax
import numpy as np
from crazyflow.control import Control
from crazyflow.control.control import state2attitude
from crazyflow.sim import Sim
from scipy.spatial.transform import Rotation as RotLib

from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make

jit_state2attitude = jax.jit(state2attitude)


def control(obs, t, i_error, dt):
    pos, vel, quat = obs.pos, obs.vel, obs.quat
    des_pos = np.array([np.cos(t) - 1, np.sin(t), 0.2 * t])
    cmd, i_error = state2attitude(
        pos, vel, quat, des_pos, np.zeros((1, 1, 3)), np.zeros((1, 1, 1)), i_error, dt
    )
    return cmd, i_error


def run(duration=5.0, fps=60, safety_filter=None):
    # Create the simulation environment.
    sim = Sim(
        n_drones=1,
        control=Control.attitude,
        attitude_freq=25,
        integrator='rk4',
        physics='sys_id',
    )
    sim.reset()
    dt = 1 / sim.control_freq

    # Run the simulation.
    i_error = np.zeros((1, 1, 3))
    for i in range(int(duration * sim.control_freq)):
        # Get the current state.
        obs = sim.data.states
        rpy = RotLib.from_quat(obs.quat.flatten()).as_euler('xyz').reshape(1, 1, 3)
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpy, obs.ang_vel], axis=-1).flatten()
        stacked_obs = stacked_obs[np.array([0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11])]

        # Compute the control command.
        cmd, i_error = control(obs, i * dt, i_error, dt)
        cmd, _ = safety_filter.certify_action(stacked_obs, cmd)

        # Apply the control command.
        sim.attitude_control(cmd.reshape(1, 1, -1))
        sim.step(sim.freq // sim.control_freq)
        if ((i * fps) % sim.control_freq) < fps:
            sim.render()
    sim.close()


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
                         **config.sf_config)
    safety_filter.reset()

    run(duration=5.0, fps=60, safety_filter=safety_filter)


if __name__ == '__main__':
    main()
