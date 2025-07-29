import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_results(num_drones, results, X_goal):
    # Extract position data
    positions = np.array([obs.pos.squeeze() for obs in results]).reshape((-1, num_drones, 3))
    x = positions[:, :, 0]
    y = positions[:, :, 1]
    z = positions[:, :, 2]

    # Get constraint bounds from safety filter config
    x_bounds = [-1, 1]
    y_bounds = [-1, 1]
    z_bounds = [0.01, 3]

    # Plot trajectory and constraints
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for drone_idx in range(positions.shape[1]):
        ax.plot(x[:, drone_idx], y[:, drone_idx], z[:, drone_idx], label=f'Trajectory_{drone_idx}', color=colors[drone_idx])
        ax.plot(X_goal[:, drone_idx, 0], X_goal[:, drone_idx, 2], X_goal[:, drone_idx, 4], label=f'Goal_{drone_idx}', color=colors[drone_idx], linestyle='--')

    add_box_to_plot(ax, x_bounds, y_bounds, z_bounds)

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

    # Plot velocity data
    # Extract velocity data
    velocities = np.array([obs.vel.squeeze() for obs in results]).reshape((-1, num_drones, 3))
    x = velocities[:, :, 0]
    y = velocities[:, :, 1]
    z = velocities[:, :, 2]

    # Get constraint bounds from safety filter config
    x_bounds = [-2, 2]
    y_bounds = [-2, 2]
    z_bounds = [-2, 2]

    # Plot trajectory and constraints
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b', 'y']
    for drone_idx in range(velocities.shape[1]):
        ax.plot(x[:, drone_idx], y[:, drone_idx], z[:, drone_idx], label=f'Velocity_{drone_idx}', color=colors[drone_idx])

    add_box_to_plot(ax, x_bounds, y_bounds, z_bounds)

    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()


def add_box_to_plot(ax, x_bounds, y_bounds, z_bounds):
    # Create data for the box faces
    xx, yy = np.meshgrid([x_bounds[0], x_bounds[1]], [y_bounds[0], y_bounds[1]])
    z1 = np.ones_like(xx) * z_bounds[0]
    z2 = np.ones_like(xx) * z_bounds[1]

    # Plot the 6 faces of the box
    ax.plot_surface(xx, yy, z1, alpha=0.1, color='r')  # Bottom
    ax.plot_surface(xx, yy, z2, alpha=0.1, color='r')  # Top

    yy, zz = np.meshgrid([y_bounds[0], y_bounds[1]], [z_bounds[0], z_bounds[1]])
    x1 = np.ones_like(yy) * x_bounds[0]
    x2 = np.ones_like(yy) * x_bounds[1]
    ax.plot_surface(x1, yy, zz, alpha=0.1, color='r')  # Left
    ax.plot_surface(x2, yy, zz, alpha=0.1, color='r')  # Right

    xx, zz = np.meshgrid([x_bounds[0], x_bounds[1]], [z_bounds[0], z_bounds[1]])
    y1 = np.ones_like(xx) * y_bounds[0]
    y2 = np.ones_like(xx) * y_bounds[1]
    ax.plot_surface(xx, y1, zz, alpha=0.1, color='r')  # Front
    ax.plot_surface(xx, y2, zz, alpha=0.1, color='r')  # Back


def create_video(frames, fps, name):
    size = 480, 640
    out = cv2.VideoWriter(f'./results/videos/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for frame in frames:
        out.write(frame)
    out.release()


def generate_X_goal(traj_type, start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    if traj_type == 'no_collision':
        return generate_no_collision_traj(start_pos, num_iters, dt)
    elif traj_type == 'mild_collision':
        return generate_mild_collision_traj(num_drones, num_iters, dt)
    elif traj_type == 'medium_collision':
        return generate_medium_collision_traj(num_drones, num_iters, dt)
    elif traj_type == 'severe_collision':
        return generate_severe_collision_traj(num_drones, num_iters, dt)


def generate_no_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    X_goal = np.zeros((num_iters, num_drones, 12))

    for i in range(num_iters):
        # Gradually increase radius from 0 to 0.5 over the trajectory
        radius = min(0.5, i / num_iters)

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

    for i in range(num_iters):
        t = i * dt

        drone0_pos = center + amplitude * np.array([
            np.cos(-t),  # x
            np.cos(-t),  # y
            np.sin(-t)  # z
        ])

        drone1_pos = center + amplitude * np.array([
            np.cos(-t),  # x
            -np.cos(-t),  # y
            np.sin(-t)  # z
        ])

        drone2_pos = center + amplitude * np.array([
            np.cos(t),  # x
            np.cos(t),  # y
            np.sin(t)  # z
        ])

        drone3_pos = center + amplitude * np.array([
            np.cos(t),  # x
            -np.cos(t),  # y
            np.sin(t)  # z
        ])

        all_pos = np.stack([drone0_pos, drone1_pos, drone2_pos, drone3_pos])
        X_goal[i, :, [0, 2, 4]] = all_pos.T

    return X_goal


def generate_severe_collision_traj(num_drones, num_iters, dt):
    assert num_drones == 4, '[ERROR] Only 4 drones are supported for this trajectory'
    return generate_4_figure_8_traj(num_drones, num_iters, dt, phase=[np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2])


def generate_4_figure_8_traj(num_drones, num_iters, dt, phase):
    X_goal = np.zeros((num_iters, num_drones, 12))

    amplitude = 1.0  # Size of the figure 8
    center = np.array([0, 0, 1.5])  # Center point of intersection

    for i in range(num_iters):
        t = i * dt

        drone0_pos = center + amplitude * np.array([
            np.sin(t + phase[0]),  # x
            np.sin(t + phase[0]) * np.cos(t + phase[0]),  # y
            np.sin(2 * (t + phase[0]))  # z
        ])

        drone1_pos = center + amplitude * np.array([
            -np.sin(t + phase[1]) * np.cos(t + phase[1]),  # x
            -np.sin(t + phase[1]),  # y
            np.sin(2 * (t + phase[1]))  # z
        ])

        drone2_pos = center + amplitude * np.array([
            np.sin(t + phase[2]) * np.cos(t + phase[2]),  # x
            np.sin(t + phase[2]),  # y
            np.sin(2 * (t + phase[2]))  # z
        ])

        drone3_pos = center + amplitude * np.array([
            -np.sin(t + phase[3]),  # x
            -np.sin(t + phase[3]) * np.cos(t + phase[3]),  # y
            np.sin(2 * (t + phase[3]))  # z
        ])

        all_pos = np.stack([drone0_pos, drone1_pos, drone2_pos, drone3_pos])
        X_goal[i, :, [0, 2, 4]] = all_pos.T

    return X_goal


def calculate_RMSE(all_obs, X_goal):
    all_pos = np.array([obs.pos[0, :, :].flatten() for obs in all_obs])
    X_goal = X_goal[:, :, [0, 2, 4]].reshape((len(all_obs), -1))
    RMSE = np.sqrt(np.mean(np.sum((all_pos - X_goal)**2, axis=1)))
    return RMSE


def calculate_constraint_violations(all_stacked_obs, constraint_bounds, num_drones):
    all_stacked_obs = np.array(all_stacked_obs)
    constraint_violations = np.zeros(all_stacked_obs.shape[1])
    for i in range(all_stacked_obs.shape[0]):
        constraint_violations += all_stacked_obs[i] - np.tile(constraint_bounds.upper_bounds, num_drones) > 0
        constraint_violations += np.tile(constraint_bounds.lower_bounds, num_drones) - all_stacked_obs[i] > 0
    return constraint_violations.reshape((num_drones, -1))


def calculate_collisions(all_obs, num_drones, min_collision_distance):
    collisions = 0
    for iter in range(len(all_obs)):
        for d1 in range(num_drones):
            for d2 in range(d1 + 1, num_drones):
                if np.linalg.norm(all_obs[iter].pos[0, d1, :] - all_obs[iter].pos[0, d2, :]) < min_collision_distance:
                    collisions += 1
    return collisions
