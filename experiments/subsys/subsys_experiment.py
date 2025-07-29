import time
from functools import partial

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation as RotLib

from experiments.subsys.subsys_utils import (calculate_collisions, calculate_constraint_violations,
                                             calculate_RMSE, create_video, generate_X_goal, plot_results)
from safe_control_gym.safety_filters.mpsc.mpsc_utils import get_discrete_derivative
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(
    gui=False,
    num_drones=1,
    duration=5.0,
    safety_filter=None,
    controller=None,
    sf_vec=None,
    traj_type='no_collision',
    sf_type='none',
):
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

    full_X_goal = generate_X_goal(traj_type, sim.data.states.pos[0, :, :], int(duration * sim.control_freq), dt)
    X_goal = full_X_goal.reshape((int(duration * sim.control_freq), num_drones * 12))
    safety_filter.env.X_GOAL = X_goal
    controller.env.X_GOAL = X_goal
    safety_filter.cost_function.uncertified_controller = controller

    start_states = full_X_goal[0, :, [0, 2, 4]].T.reshape((1, num_drones, 3))
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=start_states)
    )

    # Run the simulation.
    all_obs = []
    all_stacked_obs = []
    all_actions = []
    all_corrections = []
    start_time = time.time()
    frames = []
    cam_config = {
        'distance': 4.0,       # Distance from target to camera (increased from default)
        'elevation': -20,      # Camera elevation angle (less steep downward angle)
        'azimuth': 45,         # Camera azimuth (horizontal rotation)
        'lookat': [0, 0, 1.5]  # Look at point 1.5m above ground level
    }
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
        all_stacked_obs.append(stacked_obs)

        # Compute the control command.
        uncert_cmd = controller.select_action(stacked_obs, info={'current_step': i})
        if sf_type == 'none':
            cert_cmd = uncert_cmd
        else:
            cert_cmd, _ = safety_filter.certify_action(stacked_obs, uncert_cmd.flatten(), info={'current_step': i})
        all_actions.append(cert_cmd)

        all_corrections.append(np.linalg.norm(uncert_cmd.reshape(num_drones, -1)[sf_vec, :] - cert_cmd.reshape(num_drones, -1)[sf_vec, :]))

        # Apply the control command.
        sim.attitude_control(cert_cmd.reshape(1, num_drones, -1))
        sim.step(sim.freq // sim.control_freq)
        if i == 0:
            start_time = time.time()
        if gui:
            frames.append(sim.render(mode='rgb_array', default_cam_config=cam_config))
    print(f'Time taken: {time.time() - start_time} seconds')
    sim.close()

    print('Mean Correction:', np.round(np.mean(all_corrections), 3))
    print('Max Correction:', np.round(np.max(all_corrections), 3))

    # Print Metrics
    RMSE = calculate_RMSE(all_obs, full_X_goal)
    state_constraint_violation = calculate_constraint_violations(all_stacked_obs, safety_filter.state_constraint, num_drones)
    input_constraint_violation = calculate_constraint_violations(all_actions, safety_filter.input_constraint, num_drones)
    collisions = calculate_collisions(all_obs, num_drones, safety_filter.min_collision_distance)
    discrete_derivative = get_discrete_derivative(np.array(all_actions), sim.control_freq)
    input_rate_of_change = np.linalg.norm(discrete_derivative, 'fro')

    print('RMSE:', np.round(RMSE, 3))
    print('State Constraint Violation:', np.round(state_constraint_violation, 3))
    print('Input Constraint Violation:', np.round(input_constraint_violation, 3))
    print('Collisions:', collisions)
    print('Input Rate of Change:', np.round(input_rate_of_change, 3))

    if gui:
        create_video(frames, sim.control_freq, name=traj_type)
    plot_results(num_drones, all_obs, full_X_goal)


def main():
    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    traj_type = config.traj_type
    sf_type = config.sf_type

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
    sf_vec = [False, True, False, True]
    assert len(sf_vec) == config.num_drones
    safety_filter = make(config.safety_filter,
                         env_func,
                         num_drones=config.num_drones,
                         sf_vec=sf_vec,
                         sf_type=sf_type,
                         **config.sf_config)
    safety_filter.reset()

    run(
        gui=False,
        num_drones=config.num_drones,
        sf_vec=sf_vec,
        duration=15.0,
        safety_filter=safety_filter,
        controller=lqr_controller,
        traj_type=traj_type,
        sf_type=sf_type,
    )


if __name__ == '__main__':
    main()
