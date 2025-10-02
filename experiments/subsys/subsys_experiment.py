import pickle
import time
from functools import partial

import numpy as np
from crazyflow.control import Control
from crazyflow.sim import Sim
from munch import munchify
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

from experiments.subsys.plotting_results import plot_trajectory_3D
from experiments.subsys.subsys_utils import (calculate_collisions, calculate_constraint_violations,
                                             calculate_input_rate_of_change, calculate_open_loop_traj,
                                             calculate_RMSE, generate_lqr_gains, generate_X_goal)
from safe_control_gym.safety_filters.mpsc.mpsc_cost_function.precomputed_cost import PRECOMPUTED_COST
from safe_control_gym.utils.configuration import ConfigFactory
from safe_control_gym.utils.registration import make


def run(
    sim,
    plot=False,
    trial=0,
    frequency=25,
    duration=5.0,
    safety_filter=None,
    teleop_controller=None,
    swarm_controller=None,
    teleop_vec=None,
    X_goal=None,
    sf_type='none',
):
    sim.reset()
    num_drones = len(teleop_vec)
    nx, nu = safety_filter.model.nx, safety_filter.model.nu

    experiment_len = int(duration * sim.control_freq)
    goal_len = experiment_len + safety_filter.horizon + 1

    if sf_type in ['naive', 'safe_swarm_basic', 'safe_swarm_advanced']:
        safety_filter.env.X_GOAL = X_goal[:, teleop_vec, :].reshape((goal_len, sum(teleop_vec) * nx))
    elif sf_type != 'none':
        safety_filter.env.X_GOAL = X_goal.reshape((goal_len, num_drones * nx))

    if teleop_controller is not None:
        trial_gain = np.load('./parameters/lqr_gains.npy')[trial, :, :]
        full_gain = block_diag(*[trial_gain] * sum(teleop_vec))
        teleop_controller.gain = full_gain
        teleop_controller.env.X_GOAL = X_goal[:, teleop_vec, :].reshape((goal_len, sum(teleop_vec) * nx))
    if swarm_controller is not None:
        if sf_type in ['safe_swarm_basic', 'safe_swarm_advanced']:
            swarm_controller.env.X_GOAL = X_goal.reshape((goal_len, num_drones * nx))
        else:
            swarm_controller.env.X_GOAL = X_goal[:, ~teleop_vec, :].reshape((goal_len, sum(~teleop_vec) * nx))

    starting_position = np.load('./parameters/starting_positions.npy')[[trial], :, :]
    starting_velocity = np.load('./parameters/starting_velocities.npy')[[trial], :, :]
    sim.data = sim.data.replace(
        states=sim.data.states.replace(pos=starting_position, vel=starting_velocity)
    )

    # Run the simulation.
    all_obs = []
    all_actions = []
    all_corrections = []
    start_time = time.time()
    for step_idx in range(experiment_len):
        # Get the current state.
        obs = sim.data.states
        rpys = []
        for drone_idx in range(num_drones):
            rpy = Rotation.from_quat(obs.quat[0, drone_idx, :].flatten()).as_euler('xyz')
            rpys.append(rpy)
        rpys = np.array(rpys).reshape((1, num_drones, 3))
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpys, obs.ang_vel], axis=-1)[0, :, :]
        if nx == 10:
            stacked_obs = stacked_obs[:, [0, 3, 1, 4, 2, 5, 6, 7, 9, 10]]
        elif nx == 12:
            stacked_obs = stacked_obs[:, [0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11]]
        all_obs.append(stacked_obs)

        # Compute the control command.
        uncert_cmd = np.zeros((num_drones, nu))
        teleop_uncert_traj = np.zeros((safety_filter.horizon, sum(teleop_vec), nu))
        swarm_uncert_traj = np.zeros((safety_filter.horizon, sum(~teleop_vec), nu))
        if teleop_controller is not None:
            uncert_cmd_teleop = teleop_controller.select_action(stacked_obs[teleop_vec, :].flatten(), info={'current_step': step_idx})
            uncert_cmd[teleop_vec, :] = uncert_cmd_teleop.copy().reshape(sum(teleop_vec), nu)
            if (sf_type != 'none' and isinstance(safety_filter.cost_function, PRECOMPUTED_COST)) or sf_type == 'safe_teleop_advanced':
                teleop_uncert_traj = calculate_open_loop_traj(stacked_obs.copy(), teleop_controller, sim, teleop_vec, safety_filter.horizon, start_step=step_idx, nu=nu)
                assert np.linalg.norm(teleop_uncert_traj[0, :, :].flatten() - uncert_cmd_teleop) < 1e-6, '[ERROR] LQR trajectory and uncert_cmd_teleop are not the same.'
                safety_filter.teleop_uncert_traj = teleop_uncert_traj
        if swarm_controller is not None and sf_type not in ['safe_swarm_basic', 'safe_swarm_advanced']:
            uncert_cmd_swarm = swarm_controller.select_action(stacked_obs[~teleop_vec, :].flatten(), info={'current_step': step_idx})
            uncert_cmd[~teleop_vec, :] = uncert_cmd_swarm.copy().reshape(sum(~teleop_vec), nu)
            if sf_type == 'safe_teleop_advanced':
                swarm_uncert_traj = swarm_controller.v_prev.T[:, :].reshape(safety_filter.horizon, sum(~teleop_vec), nu)
                assert np.linalg.norm(swarm_uncert_traj[0, :, :].flatten() - uncert_cmd_swarm) < 1e-6, '[ERROR] Swarm trajectory and uncert_cmd_swarm are not the same.'
                safety_filter.swarm_uncert_traj = swarm_uncert_traj
        uncert_cmd = uncert_cmd.flatten()

        if sf_type == 'none' or sum(teleop_vec) == 0:
            cert_cmd = uncert_cmd
        else:
            if sf_type in ['naive', 'safe_swarm_basic', 'safe_swarm_advanced']:
                cert_cmd, _ = safety_filter.certify_action(
                    stacked_obs.reshape(num_drones, nx)[teleop_vec, :].flatten(),
                    uncert_cmd.reshape(num_drones, nu)[teleop_vec, :].flatten(),
                    info={'current_step': step_idx})
            else:
                cert_cmd, _ = safety_filter.certify_action(stacked_obs.flatten(), uncert_cmd.flatten(), info={'current_step': step_idx})

            if sf_type in ['naive', 'safe_swarm_basic', 'safe_swarm_advanced']:
                full_cert_cmd = uncert_cmd.reshape(num_drones, nu).copy()
                full_cert_cmd[teleop_vec, :] = cert_cmd.reshape(sum(teleop_vec), nu)
                cert_cmd = full_cert_cmd.flatten()

        if swarm_controller is not None and sf_type in ['safe_swarm_basic', 'safe_swarm_advanced']:
            cert_traj = np.tile(swarm_controller.model.U_EQ, (swarm_controller.horizon, sum(teleop_vec), 1))
            cert_traj[:, :, :] = safety_filter.v_prev.T[:, :].reshape(safety_filter.horizon, sum(teleop_vec), nu)
            assert np.linalg.norm(cert_traj[0, :, :] - full_cert_cmd[teleop_vec, :]) < 1e-6, '[ERROR] SF trajectory and cert_cmd are not the same.'
            swarm_controller.teleop_uncert_traj = cert_traj
            cert_cmd_swarm = swarm_controller.select_action(stacked_obs.flatten(), info={'current_step': step_idx})
            cert_cmd = cert_cmd.copy().reshape(num_drones, nu)
            cert_cmd[~teleop_vec, :] = cert_cmd_swarm.reshape(len(teleop_vec), nu)[~teleop_vec, :]
            cert_cmd = cert_cmd.flatten()

        all_actions.append(cert_cmd.reshape(num_drones, nu))
        all_corrections.append(np.linalg.norm(uncert_cmd.reshape(num_drones, nu)[teleop_vec, :] - cert_cmd.reshape(num_drones, nu)[teleop_vec, :]))

        cert_cmd_clipped = np.clip(cert_cmd.copy(),
                                   np.tile(safety_filter.input_constraint.lower_bounds, num_drones),
                                   np.tile(safety_filter.input_constraint.upper_bounds, num_drones))

        if nx == 10:
            cert_cmd_clipped = np.hstack((cert_cmd_clipped.reshape(num_drones, nu), np.zeros((num_drones, 1)))).reshape((1, num_drones, nu + 1))
        elif nx == 12:
            cert_cmd_clipped = cert_cmd_clipped.reshape((1, num_drones, nu))

        # Apply the control command.
        sim.attitude_control(cert_cmd_clipped)
        sim.step(sim.freq // sim.control_freq)
        if step_idx == 2:  # First few iters are very slow, so we don't count them
            start_time = time.time()

    time_taken = time.time() - start_time

    all_obs = np.array(all_obs)
    all_actions = np.array(all_actions)
    all_corrections = np.array(all_corrections)

    all_results = munchify({
        'sf_type': sf_type,
        'teleop_vec': teleop_vec,
        'X_goal': X_goal,
        'obs': all_obs,
        'actions': all_actions,
        'corrections': all_corrections,
        'state_constraints': {
            'lower_bounds': safety_filter.state_constraint.lower_bounds,
            'upper_bounds': safety_filter.state_constraint.upper_bounds,
        },
        'input_constraints': {
            'lower_bounds': safety_filter.input_constraint.lower_bounds,
            'upper_bounds': safety_filter.input_constraint.upper_bounds,
        },
        'min_collision_distance': safety_filter.min_collision_distance,
        'experiment_len': experiment_len,
        'frequency': frequency,
        'time': time_taken,
    })

    # Print Metrics
    RMSE = calculate_RMSE(all_obs, X_goal[:experiment_len, :, :])
    state_constraint_violation = calculate_constraint_violations(all_obs, safety_filter.state_constraint)
    input_constraint_violation = calculate_constraint_violations(all_actions, safety_filter.input_constraint)
    collisions = calculate_collisions(all_obs, safety_filter.min_collision_distance)
    input_rate_of_change = calculate_input_rate_of_change(all_actions, frequency)

    print(f'Time taken: {time_taken} seconds')
    print('Mean Correction:', np.round(np.mean(all_corrections), 3))
    print('Max Correction:', np.round(np.max(all_corrections), 3))
    print('RMSE:', np.round(RMSE, 3))
    print('State Constraint Violation:', np.round(state_constraint_violation, 3))
    print('Input Constraint Violation:', np.round(input_constraint_violation, 3))
    print('Collisions:', collisions)
    print('Input Rate of Change:', np.round(input_rate_of_change, 3))

    if plot:
        plot_trajectory_3D(all_obs, X_goal, safety_filter.state_constraint)

    return all_results


def main(gen_lqr_gains=False):
    # The SF vector is used to select the drones that are certified.
    teleop_vec = np.array([False, True, False, True])
    num_trials = 4

    # Create the configuration dictionary.
    fac = ConfigFactory()
    config = fac.merge()
    sf_type = config.sf_type
    if sf_type == 'none_lqr':
        teleop_vec = np.array([True, True, True, True])
        sf_type = 'none'
    elif sf_type == 'none_mpc':
        teleop_vec = np.array([False, False, False, False])
        sf_type = 'none'

    # Generate goal trajectory.
    frequency = config.task_config.ctrl_freq
    duration = 15.0

    if config.task_config.quad_type == 8:
        nx = 10
        nu = 3
    elif config.task_config.quad_type == 6:
        nx = 12
        nu = 4
    else:
        raise ValueError(f'Quad type {config.task_config.quad_type} not supported.')

    X_goal = generate_X_goal(config.traj_type, int(duration * frequency) + config.sf_config.horizon + 1, 1 / frequency, nx)

    # Create a cost function.
    cost_func = config.sf_config.cost_function
    mpsc_cost_horizon = config.sf_config.mpsc_cost_horizon
    decay_factor = config.sf_config.decay_factor
    del config.sf_config.cost_function
    del config.sf_config.mpsc_cost_horizon
    del config.sf_config.decay_factor

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
        if gen_lqr_gains:
            generate_lqr_gains(num_trials, lqr_controller.model, config.algo_config.q_lqr, config.algo_config.r_lqr)
            raise SystemExit
        lqr_controller.model.U_EQ = np.tile(lqr_controller.model.U_EQ, sum(teleop_vec))

    # Create an MPC controller
    if sum(~teleop_vec) == 0 or sf_type == 'ours':
        mpc_controller = None
    else:
        if sf_type in ['safe_swarm_basic', 'safe_swarm_advanced']:
            mpc_controller = make(config.safety_filter,
                                  env_func,
                                  initial_state=X_goal[0, :, :],
                                  teleop_vec=teleop_vec,
                                  sf_type=sf_type,
                                  mpc_mode=True,
                                  cost_function='precomputed_cost',
                                  mpsc_cost_horizon=20,
                                  decay_factor=1.0,
                                  **config.sf_config,
                                  )
        else:
            mpc_controller = make(config.safety_filter,
                                  env_func,
                                  initial_state=X_goal[0, ~teleop_vec, :],
                                  teleop_vec=np.array([False] * sum(~teleop_vec)),
                                  sf_type='ours',
                                  mpc_mode=True,
                                  cost_function='one_step_cost',
                                  mpsc_cost_horizon=mpsc_cost_horizon,
                                  decay_factor=decay_factor,
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
            'model': {
                'nx': nx,
                'nu': nu,
            },
        }
        safety_filter = munchify(safety_filter)
    else:
        if sf_type in ['naive', 'safe_swarm_basic', 'safe_swarm_advanced']:
            safety_filter = make(config.safety_filter,
                                 env_func,
                                 initial_state=X_goal[0, teleop_vec, :],
                                 teleop_vec=np.array([True] * sum(teleop_vec)),
                                 sf_type='naive',
                                 cost_function=cost_func,
                                 mpsc_cost_horizon=mpsc_cost_horizon,
                                 decay_factor=decay_factor,
                                 **config.sf_config)
        else:
            safety_filter = make(config.safety_filter,
                                 env_func,
                                 initial_state=X_goal[0, :, :],
                                 teleop_vec=teleop_vec,
                                 sf_type=sf_type,
                                 cost_function=cost_func,
                                 mpsc_cost_horizon=mpsc_cost_horizon,
                                 decay_factor=decay_factor,
                                 **config.sf_config)

        safety_filter.reset()

    # Create the simulation environment.
    sim = Sim(
        n_drones=len(teleop_vec),
        control=Control.attitude,
        attitude_freq=frequency,
        integrator='rk4',
        physics='sys_id',
    )
    all_results = []
    for trial in range(num_trials):
        print(f'\nRunning trial {trial + 1} of {num_trials}. SF_TYPE: {sf_type}.')
        print('--------------------------------')
        results = run(
            sim=sim,
            plot=False,
            trial=trial,
            teleop_vec=teleop_vec,
            frequency=frequency,
            duration=duration,
            safety_filter=safety_filter,
            teleop_controller=lqr_controller,
            swarm_controller=mpc_controller,
            X_goal=X_goal,
            sf_type=sf_type,
        )
        all_results.append(results)
    sim.close()

    # Save results to pickle file
    if sf_type == 'none' and sum(teleop_vec) == 4:
        name = 'none_lqr'
    elif sf_type == 'none' and sum(teleop_vec) == 0:
        name = 'none_mpc'
    else:
        name = sf_type

    with open(f'./results/experiments/{config.traj_type}/{"multi_trial/" if num_trials > 1 else ""}{name}.pkl', 'wb') as f:
        pickle.dump(all_results, f)


if __name__ == '__main__':
    main(gen_lqr_gains=False)
