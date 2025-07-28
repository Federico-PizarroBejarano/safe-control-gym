import cv2
import matplotlib.pyplot as plt
import numpy as np


def plot_results(num_drones, results):
    # Extract position data
    positions = np.array([obs.pos.squeeze() for obs in results]).reshape((-1, num_drones, 3))
    x = positions[:, :, 0]
    y = positions[:, :, 1]

    # Get constraint bounds from safety filter config
    x_bounds = [-1, 1]
    y_bounds = [-1, 1]

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


def create_video(frames, fps, name):
    size = 480, 640
    out = cv2.VideoWriter(f'./results/videos/{name}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
    for frame in frames:
        out.write(frame)
    out.release()


def generate_X_goal(traj_type, start_pos, num_iters, dt):
    if traj_type == 'no_collision':
        return generate_no_collision_traj(start_pos, num_iters, dt)
    elif traj_type == 'mild_collision':
        return generate_mild_collision_traj(start_pos, num_iters, dt)
    elif traj_type == 'medium_collision':
        return generate_medium_collision_traj(start_pos, num_iters, dt)
    elif traj_type == 'severe_collision':
        return generate_severe_collision_traj(start_pos, num_iters, dt)


def generate_no_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    X_goal = np.zeros((num_iters, num_drones, 12))

    for i in range(num_iters):
        # Gradually increase radius from 0 to 0.5 over the trajectory
        radius = min(0.5, i / num_iters)

        spiral_x = start_pos[:, 0] + radius * np.cos(i * dt)
        spiral_y = start_pos[:, 1] + radius * np.sin(i * dt)
        spiral_z = start_pos[:, 2] + radius

        X_goal[i, :, [0, 2, 4]] = np.stack([spiral_x, spiral_y, spiral_z], axis=1).T

    return X_goal


def generate_mild_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    assert num_drones == 4, '[ERROR] Only 4 drones are supported for this trajectory'
    return generate_4_figure_8_traj(start_pos, num_iters, dt, phase=[0, 0, np.pi / 2, np.pi / 2])


def generate_medium_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
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


def generate_severe_collision_traj(start_pos, num_iters, dt):
    num_drones = start_pos.shape[0]
    assert num_drones == 4, '[ERROR] Only 4 drones are supported for this trajectory'
    return generate_4_figure_8_traj(start_pos, num_iters, dt, phase=[0, 0, 0, 0])


def generate_4_figure_8_traj(start_pos, num_iters, dt, phase):
    num_drones = start_pos.shape[0]
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
