import time
from functools import partial

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from munch import munchify
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from experiments.subsys.subsys_utils import (calculate_collisions, calculate_constraint_violations,
                                             calculate_input_rate_of_change, calculate_open_loop_traj,
                                             calculate_RMSE, create_video, generate_X_goal, plot_results)
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.precomputed_cost import PRECOMPUTED_COST
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(
    gui=False,
    num_drones=1,
    frequency=25,
    duration=5.0,
    safety_filter=None,
    teleop_controller=None,
    swarm_controller=None,
    teleop_vec=None,
    X_goal=None,
    sf_type='none',
):
    # Create the simulation environment.
    sim = Sim(
        n_drones=num_drones,
        control=Control.attitude,
        attitude_freq=frequency,
        integrator='rk4',
        physics='sys_id',
    )
    sim.reset()

    experiment_len = int(duration * sim.control_freq)
    goal_len = experiment_len + safety_filter.horizon + 1

    if sf_type == 'naive':
        safety_filter.env.X_GOAL = X_goal[:, teleop_vec, :].reshape((goal_len, sum(teleop_vec) * 12))
    elif sf_type != 'none':
        safety_filter.env.X_GOAL = X_goal.reshape((goal_len, num_drones * 12))

    if teleop_controller is not None:
        teleop_controller.env.X_GOAL = X_goal[:, teleop_vec, :].reshape((goal_len, sum(teleop_vec) * 12))
    if swarm_controller is not None:
        swarm_controller.env.X_GOAL = X_goal[:, ~teleop_vec, :].reshape((goal_len, sum(~teleop_vec) * 12))

    start_pos = X_goal[0, :, [0, 2, 4]].T.reshape((1, num_drones, 3))
    start_vel = X_goal[0, :, [1, 3, 5]].T.reshape((1, num_drones, 3))
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=start_pos, vel=start_vel)
    )

    # Run the simulation.
    all_obs = []
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
    for i in range(experiment_len):
        # Get the current state.
        obs = sim.data.states
        rpys = []
        for drone_idx in range(num_drones):
            rpy = Rotation.from_quat(obs.quat[0, drone_idx, :].flatten()).as_euler('xyz')
            rpys.append(rpy)
        rpys = np.array(rpys).reshape((1, num_drones, 3))
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpys, obs.ang_vel], axis=-1)[0, :, :]
        stacked_obs = stacked_obs[:, [0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11]]
        all_obs.append(stacked_obs)

        # Compute the control command.
        uncert_cmd = np.zeros((num_drones, 4))
        uncert_traj = np.zeros((safety_filter.horizon, num_drones, 4))
        if teleop_controller is not None:
            uncert_cmd_teleop = teleop_controller.select_action(stacked_obs[teleop_vec, :].flatten(), info={'current_step': i})
            uncert_cmd[teleop_vec, :] = uncert_cmd_teleop.copy().reshape(sum(teleop_vec), 4)
            if (sf_type != 'none' and isinstance(safety_filter.cost_function, PRECOMPUTED_COST)) or sf_type == 'safe_teleop_advanced':
                lqr_trajectory = calculate_open_loop_traj(stacked_obs.copy(), teleop_controller, sim, teleop_vec, safety_filter.horizon, start_step=i)
                uncert_traj[:, teleop_vec, :] = lqr_trajectory
                assert np.linalg.norm(uncert_traj[0, teleop_vec, :].flatten() - uncert_cmd_teleop) < 1e-6, '[ERROR] LQR trajectory and uncert_cmd_teleop are not the same.'
        if swarm_controller is not None:
            uncert_cmd_swarm = swarm_controller.select_action(stacked_obs[~teleop_vec, :].flatten(), info={'current_step': i})
            uncert_cmd[~teleop_vec, :] = uncert_cmd_swarm.copy().reshape(sum(~teleop_vec), 4)
            uncert_traj[:, ~teleop_vec, :] = swarm_controller.v_prev.T[:, :].reshape(safety_filter.horizon, sum(~teleop_vec), 4)
            assert np.linalg.norm(uncert_traj[0, ~teleop_vec, :].flatten() - uncert_cmd_swarm) < 1e-6, '[ERROR] Swarm trajectory and uncert_cmd_swarm are not the same.'
        uncert_cmd = uncert_cmd.flatten()

        if sf_type == 'none' or sum(teleop_vec) == 0:
            cert_cmd = uncert_cmd
        else:
            safety_filter.uncert_traj = uncert_traj
            if sf_type == 'naive':
                uncert_traj = uncert_traj[:, teleop_vec, :]
                safety_filter.uncert_traj = uncert_traj
                cert_cmd, _ = safety_filter.certify_action(
                    stacked_obs.reshape(num_drones, -1)[teleop_vec, :].flatten(),
                    uncert_cmd.reshape(num_drones, -1)[teleop_vec, :].flatten(),
                    info={'current_step': i})
            else:
                cert_cmd, _ = safety_filter.certify_action(stacked_obs.flatten(), uncert_cmd.flatten(), info={'current_step': i})

            if sf_type == 'naive':
                full_cert_cmd = uncert_cmd.reshape(num_drones, -1).copy()
                full_cert_cmd[teleop_vec, :] = cert_cmd.reshape(sum(teleop_vec), -1)
                cert_cmd = full_cert_cmd.flatten()

        all_actions.append(cert_cmd.reshape(num_drones, -1))
        all_corrections.append(np.linalg.norm(uncert_cmd.reshape(num_drones, -1)[teleop_vec, :] - cert_cmd.reshape(num_drones, -1)[teleop_vec, :]))

        cert_cmd_clipped = np.clip(cert_cmd.copy(),
                                   np.tile(safety_filter.input_constraint.lower_bounds, num_drones),
                                   np.tile(safety_filter.input_constraint.upper_bounds, num_drones))

        # Apply the control command.
        sim.attitude_control(cert_cmd_clipped.reshape(1, num_drones, -1))
        sim.step(sim.freq // sim.control_freq)
        if i == 2:  # First few iters are very slow, so we don't count them
            start_time = time.time()
        if gui:
            frames.append(sim.render(mode='rgb_array', default_cam_config=cam_config))

    print(f'Time taken: {time.time() - start_time} seconds')
    sim.close()

    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    all_corrections = np.array(all_corrections)

    print('Mean Correction:', np.round(np.mean(all_corrections), 3))
    print('Max Correction:', np.round(np.max(all_corrections), 3))

    # Print Metrics
    RMSE = calculate_RMSE(all_obs, X_goal[:experiment_len, :, :])
    state_constraint_violation = calculate_constraint_violations(all_obs, safety_filter.state_constraint)
    input_constraint_violation = calculate_constraint_violations(all_actions, safety_filter.input_constraint)
    collisions = calculate_collisions(all_obs, safety_filter.min_collision_distance)
    input_rate_of_change = calculate_input_rate_of_change(all_actions, frequency)

    print('RMSE:', np.round(RMSE, 3))
    print('State Constraint Violation:', np.round(state_constraint_violation, 3))
    print('Input Constraint Violation:', np.round(input_constraint_violation, 3))
    print('Collisions:', collisions)
    print('Input Rate of Change:', np.round(input_rate_of_change, 3))

    if gui:
        create_video(frames, sim.control_freq, sf_type)
    plot_results(all_obs, X_goal)


def main():
    # The SF vector is used to select the drones that are certified.
    teleop_vec = np.array([False, True, False, True])

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    sf_type = config.sf_type

    # Generate goal trajectory.
    frequency = config.task_config.ctrl_freq
    duration = 15.0
    X_goal = generate_X_goal(config.traj_type, int(duration * frequency) + config.sf_config.horizon + 1, 1 / frequency)

    # Create a cost function.
    cost_func = config.sf_config.cost_function
    del config.sf_config.cost_function

    # Create an environment
    env_func = partial(make,
                       config.task,
                       **config.task_config)

    # Create an LQR controller
    if sum(teleop_vec) == 0:
        lqr_controller = None
    else:
        lqr_controller = make(config.algo,
                              env_func,
                              **config.algo_config)
        lqr_controller.reset()
        lqr_controller.gain = block_diag(*[lqr_controller.gain] * sum(teleop_vec))
        lqr_controller.model.U_EQ = np.tile(lqr_controller.model.U_EQ, sum(teleop_vec))

    # Create an MPC controller
    if sum(~teleop_vec) == 0 or sf_type == 'ours':
        mpc_controller = None
    else:
        mpc_controller = make(config.safety_filter,
                              env_func,
                              initial_state=X_goal[0, ~teleop_vec, :],
                              teleop_vec=np.array([False] * sum(~teleop_vec)),
                              sf_type='ours',
                              mpc_mode=True,
                              cost_function='one_step_cost',
                              **config.sf_config,
                              )
        mpc_controller.reset()

    # Setup MPSC.
    if sf_type == 'none':
        safety_filter = {
            'horizon': config.sf_config.horizon,
            'min_collision_distance': config.sf_config.min_collision_distance,
            'state_constraint': {
                'lower_bounds': config.task_config.constraints[0].lower_bounds,
                'upper_bounds': config.task_config.constraints[0].upper_bounds,
            },
            'input_constraint': {
                'lower_bounds': config.task_config.constraints[1].lower_bounds,
                'upper_bounds': config.task_config.constraints[1].upper_bounds,
            },
        }
        safety_filter = munchify(safety_filter)
    else:
        if sf_type == 'naive':
            sf_initial_state = X_goal[0, teleop_vec, :]
        else:
            sf_initial_state = X_goal[0, :, :]
        safety_filter = make(config.safety_filter,
                             env_func,
                             initial_state=sf_initial_state,
                             teleop_vec=np.array([True] * sum(teleop_vec)) if sf_type == 'naive' else teleop_vec,
                             sf_type=sf_type,
                             cost_function=cost_func,
                             **config.sf_config)
        safety_filter.reset()

    run(
        gui=False,
        num_drones=len(teleop_vec),
        teleop_vec=teleop_vec,
        frequency=frequency,
        duration=duration,
        safety_filter=safety_filter,
        teleop_controller=lqr_controller,
        swarm_controller=mpc_controller,
        X_goal=X_goal,
        sf_type=sf_type,
    )


if __name__ == '__main__':
    main()
