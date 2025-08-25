import numpy as np
from scipy.spatial.transform import Rotation


def generate_X_goal(traj_type, num_iters, dt):
    num_drones = 4
    if traj_type == 'no_collision':
        start_pos = np.array([[0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5]])
        return generate_no_collision_traj(start_pos, num_iters, dt)
    elif traj_type == 'mild_collision':
        return generate_mild_collision_traj(num_drones, num_iters, dt)
    elif traj_type == 'medium_collision':
        return generate_medium_collision_traj(num_drones, num_iters, dt)
    elif traj_type == 'severe_collision':
        return generate_severe_collision_traj(num_drones, num_iters, dt)
    else:
        raise ValueError(f'Invalid trajectory type: {traj_type}')


def generate_no_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    X_goal = np.zeros((num_iters, num_drones, 12))

    for i in range(num_iters):
        # Gradually increase radius from 0 to 0.5 over the trajectory
        radius = min(0.6, i / num_iters)

        spiral_x = start_pos[:, 0] + radius * np.cos(i * dt)
        spiral_y = start_pos[:, 1] + radius * np.sin(i * dt)
        spiral_z = 0.1 * np.ones(num_drones) + radius

        X_goal[i, :, [0, 2, 4]] = np.stack([spiral_x, spiral_y, spiral_z], axis=1).T

    return X_goal


def generate_mild_collision_traj(num_drones, num_iters, dt):
    assert num_drones == 4, '[ERROR] Only 4 drones are supported for this trajectory'
    return generate_4_figure_8_traj(num_drones, num_iters, dt, phase=[np.pi / 4, np.pi / 4, 3 * np.pi / 4, 3 * np.pi / 4])


def generate_medium_collision_traj(num_drones, num_iters, dt):
    X_goal = np.zeros((num_iters, num_drones, 12))

    amplitude = 1.0  # Size of the figure 8
    center = np.array([0, 0, 1.5])  # Center point of intersection
    freq_mult = 0.8

    for i in range(num_iters):
        t = i * dt

        # Drone 0
        drone0_pos = center + amplitude * np.array([
            np.cos(-t * freq_mult),  # x
            np.cos(-t * freq_mult),  # y
            np.sin(-t * freq_mult)  # z
        ])
        drone0_vel = amplitude * freq_mult * np.array([
            np.sin(-t * freq_mult),  # dx/dt
            np.sin(-t * freq_mult),  # dy/dt
            np.cos(-t * freq_mult)  # dz/dt
        ])

        # Drone 1
        drone1_pos = center + amplitude * np.array([
            np.cos(-t * freq_mult),  # x
            -np.cos(-t * freq_mult),  # y
            np.sin(-t * freq_mult)  # z
        ])
        drone1_vel = amplitude * freq_mult * np.array([
            np.sin(-t * freq_mult),  # dx/dt
            -np.sin(-t * freq_mult),  # dy/dt
            np.cos(-t * freq_mult)  # dz/dt
        ])

        # Drone 2
        drone2_pos = center + amplitude * np.array([
            np.cos(t * freq_mult),  # x
            np.cos(t * freq_mult),  # y
            np.sin(t * freq_mult)  # z
        ])
        drone2_vel = amplitude * freq_mult * np.array([
            -np.sin(t * freq_mult),  # dx/dt
            -np.sin(t * freq_mult),  # dy/dt
            np.cos(t * freq_mult)  # dz/dt
        ])

        # Drone 3
        drone3_pos = center + amplitude * np.array([
            np.cos(t * freq_mult),  # x
            -np.cos(t * freq_mult),  # y
            np.sin(t * freq_mult)  # z
        ])
        drone3_vel = amplitude * freq_mult * np.array([
            -np.sin(t * freq_mult),  # dx/dt
            np.sin(t * freq_mult),  # dy/dt
            np.cos(t * freq_mult)  # dz/dt
        ])

        all_pos = np.stack([drone0_pos, drone1_pos, drone2_pos, drone3_pos])
        all_vel = np.stack([drone0_vel, drone1_vel, drone2_vel, drone3_vel])
        X_goal[i, :, [0, 2, 4]] = all_pos.T
        X_goal[i, :, [1, 3, 5]] = all_vel.T

    return X_goal


def generate_severe_collision_traj(num_drones, num_iters, dt):
    assert num_drones == 4, '[ERROR] Only 4 drones are supported for this trajectory'
    return generate_4_figure_8_traj(num_drones, num_iters, dt, phase=[np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])


def generate_4_figure_8_traj(num_drones, num_iters, dt, phase):
    X_goal = np.zeros((num_iters, num_drones, 12))

    amplitude = 1.0  # Size of the figure 8
    center = np.array([0, 0, 1.5])  # Center point of intersection
    freq_mult = 0.75

    for i in range(num_iters):
        t = i * dt

        # Drone 0
        drone0_pos = center + amplitude * np.array([
            np.sin(t * freq_mult + phase[0]),  # x
            np.sin(t * freq_mult + phase[0]) * np.cos(t * freq_mult + phase[0]),  # y
            np.sin(2 * (t * freq_mult + phase[0]))  # z
        ])
        drone0_vel = amplitude * freq_mult * np.array([
            np.cos(t * freq_mult + phase[0]),  # dx/dt
            np.cos(2 * (t * freq_mult + phase[0])),  # dy/dt
            2 * np.cos(2 * (t * freq_mult + phase[0]))  # dz/dt
        ])

        # Drone 1
        drone1_pos = center + amplitude * np.array([
            -np.sin(t * freq_mult + phase[1]) * np.cos(t * freq_mult + phase[1]),  # x
            -np.sin(t * freq_mult + phase[1]),  # y
            np.sin(2 * (t * freq_mult + phase[1]))  # z
        ])
        drone1_vel = amplitude * freq_mult * np.array([
            -np.cos(2 * (t * freq_mult + phase[1])),  # dx/dt
            -np.cos(t * freq_mult + phase[1]),  # dy/dt
            2 * np.cos(2 * (t * freq_mult + phase[1]))  # dz/dt
        ])

        # Drone 2
        drone2_pos = center + amplitude * np.array([
            np.sin(t * freq_mult + phase[2]) * np.cos(t * freq_mult + phase[2]),  # x
            np.sin(t * freq_mult + phase[2]),  # y
            np.sin(2 * (t * freq_mult + phase[2]))  # z
        ])
        drone2_vel = amplitude * freq_mult * np.array([
            np.cos(2 * (t * freq_mult + phase[2])),  # dx/dt
            np.cos(t * freq_mult + phase[2]),  # dy/dt
            2 * np.cos(2 * (t * freq_mult + phase[2]))  # dz/dt
        ])

        # Drone 3
        drone3_pos = center + amplitude * np.array([
            -np.sin(t * freq_mult + phase[3]),  # x
            -np.sin(t * freq_mult + phase[3]) * np.cos(t * freq_mult + phase[3]),  # y
            np.sin(2 * (t * freq_mult + phase[3]))  # z
        ])
        drone3_vel = amplitude * freq_mult * np.array([
            -np.cos(t * freq_mult + phase[3]),  # dx/dt
            -np.cos(2 * (t * freq_mult + phase[3])),  # dy/dt
            2 * np.cos(2 * (t * freq_mult + phase[3]))  # dz/dt
        ])

        all_pos = np.stack([drone0_pos, drone1_pos, drone2_pos, drone3_pos])
        all_vel = np.stack([drone0_vel, drone1_vel, drone2_vel, drone3_vel])
        X_goal[i, :, [0, 2, 4]] = all_pos.T
        X_goal[i, :, [1, 3, 5]] = all_vel.T

    return X_goal


def calculate_RMSE(all_obs, X_goal):
    all_pos = all_obs[:, :, [0, 2, 4]]
    X_goal_pos = X_goal[:, :, [0, 2, 4]]
    RMSEs = []
    for drone in range(all_pos.shape[1]):
        RMSEs.append(np.sqrt(np.mean(np.sum((all_pos[:, drone, :] - X_goal_pos[:, drone, :])**2, axis=1))))
    return np.array(RMSEs)


def calculate_constraint_violations(all_values, constraint_bounds):
    num_drones = all_values.shape[1]
    all_values = all_values.reshape((len(all_values), -1))
    constraint_violations = np.zeros(all_values.shape[1])
    for i in range(all_values.shape[0]):
        constraint_violations += all_values[i, :] - np.tile(constraint_bounds.upper_bounds, num_drones) > 0
        constraint_violations += np.tile(constraint_bounds.lower_bounds, num_drones) - all_values[i, :] > 0
    return constraint_violations.reshape((num_drones, -1))


def calculate_collisions(all_obs, min_collision_distance):
    num_drones = all_obs.shape[1]
    collision_matrix = np.zeros((num_drones, num_drones))
    for timestep in range(len(all_obs)):
        for d1 in range(num_drones):
            for d2 in range(d1 + 1, num_drones):
                if np.linalg.norm(all_obs[timestep, d1, [0, 2, 4]] - all_obs[timestep, d2, [0, 2, 4]]) < min_collision_distance:
                    collision_matrix[d1, d2] += 1
                    collision_matrix[d2, d1] += 1
    return collision_matrix


def calculate_input_rate_of_change(all_actions, frequency):
    num_drones = all_actions.shape[1]
    input_rate_of_change = np.zeros((num_drones))
    for i in range(num_drones):
        input_rate_of_change[i] = np.linalg.norm(np.diff(all_actions[:, i, :], axis=0), 'fro') * frequency
    return input_rate_of_change


def calculate_open_loop_traj(stacked_obs, controller, sim, teleop_vec, horizon, start_step=0):
    '''Calculate open-loop trajectory by simulating forward.

    Args:
        stacked_obs (np.ndarray): Current observation/state
        controller (Controller): Controller to generate actions
        sim (CrazyflowSimulator): Crazyflow simulator instance
        teleop_vec (np.ndarray): Safety filter vector
        horizon (int): Number of steps to simulate forward
        start_step (int): Start step for the controller

    Returns:
        input_traj (np.ndarray): Array of input commands for horizon steps
    '''
    # Save initial state
    initial_state = sim.data

    input_traj = []

    # Simulate forward for horizon steps
    for i in range(horizon):
        action = np.zeros((len(teleop_vec), 4))
        action[teleop_vec, :] = controller.select_action(stacked_obs[teleop_vec, :].flatten(), info={'current_step': i + start_step}).reshape(sum(teleop_vec), 4)
        input_traj.append(action[teleop_vec, :])

        # Step simulation
        sim.attitude_control(action.reshape((1, len(teleop_vec), 4)))
        sim.step(sim.freq // sim.control_freq)

        # Get next observation
        obs = sim.data.states
        rpys = []
        for drone_idx in range(len(teleop_vec)):
            rpy = Rotation.from_quat(obs.quat[0, drone_idx, :].flatten()).as_euler('xyz')
            rpys.append(rpy)
        rpys = np.array(rpys).reshape((1, len(teleop_vec), 3))
        stacked_obs = np.concatenate([obs.pos, obs.vel, rpys, obs.ang_vel], axis=-1)[0, :, :]
        stacked_obs = stacked_obs[:, [0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11]]

    # Reset simulator to initial state
    sim.data = initial_state

    return np.array(input_traj)
