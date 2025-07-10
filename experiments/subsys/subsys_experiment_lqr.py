import time
from functools import partial

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as RotLib

from experiments.subsys.subsys_experiment import generate_X_goal, plot_results
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(plot=False, num_drones=1, duration=5.0, fps=60, safety_filter=None, controller=None):
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
    X_goal = X_goal.reshape((int(duration * sim.control_freq), num_drones * 12))
    safety_filter.env.X_GOAL = X_goal
    controller.env.X_GOAL = X_goal
    safety_filter.cost_function.uncertified_controller = controller

    # Run the simulation.
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
        uncert_cmd = controller.select_action(stacked_obs, info={'current_step': i})
        cert_cmd, _ = safety_filter.certify_action(stacked_obs, uncert_cmd.flatten(), info={'current_step': i})
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

    print('Mean Correction:', np.mean(all_corrections))
    print('Max Correction:', np.max(all_corrections))

    plot_results(num_drones, all_obs)


def main():
    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Create an LQR controller
    lqr_controller = make(config.algo,
                          env_func,
                          **config.algo_config)
    lqr_controller.reset()
    lqr_controller.gain = block_diag(*[lqr_controller.gain] * config.num_drones)
    lqr_controller.model.U_EQ = np.tile(lqr_controller.model.U_EQ, (config.num_drones))

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
        controller=lqr_controller,
    )


if __name__ == '__main__':
    main()
