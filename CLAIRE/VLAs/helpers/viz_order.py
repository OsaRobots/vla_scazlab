import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Starting position of the robot's end effector
START_POSITION = np.array([270, 50, 150])  # (x, y, z)

def process_trajectory(data_dir, file_prefix, start_ep, ep_num, colormap, reorder=None):
    """
    Process trajectories from a dataset.
    """
    trajectories = []
    colors = colormap(np.linspace(0, 1, ep_num - start_ep))  # Unique colors for episodes in this dataset

    tm = np.array([[0, 1, 0, 0],  # [z, y, -x]
                   [0, 0, -1, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1]])

    for j in range(start_ep, ep_num):
        # Construct file path for the episode
        ep_path = os.path.join(data_dir, f'{file_prefix}{j}.npy')
        if not os.path.exists(ep_path):
            print(f"Episode file {ep_path} not found!")
            continue
        
        ep = np.load(ep_path, allow_pickle=True)
        print(f'Processing Episode {j} in {data_dir}')
        
        # Initialize the trajectory
        trajectory = [START_POSITION.copy()]  # Start at initial position
        gripper_states = []
        
        for i, step in enumerate(ep):
            if i == 0:
                # First step starts at START_POSITION
                current_position = START_POSITION
            else:
                # Compute delta and update position
                delta_position = step['action'][:3]
                if reorder:
                    delta_position = apply_inverse_transformation(step['action'],tm)
                    delta_position = delta_position[:3]
                current_position = trajectory[-1] + delta_position
            
            # Append the current position and gripper state
            trajectory.append(current_position.copy())
            gripper_states.append(step['action'][-1])  # Assuming last value in action is gripper state
        
        # Store the trajectory, gripper states, and color
        trajectories.append((trajectory, gripper_states, colors[j - start_ep]))
    
    return trajectories

def apply_inverse_transformation(data_vector, transformation_matrix):
    """
    Apply the inverse of a 4x4 transformation matrix to a data vector.
    Args:
        data_vector (np.ndarray): The transformed vector [x', y', z', r_x', r_y', r_z'] (+ gripper if is_action).
        transformation_matrix (np.ndarray): The original 4x4 transformation matrix.
        is_action (bool): Whether the input vector is an action (7 fields, with gripper as the last field).
    Returns:
        np.ndarray: Reverted vector [x, y, z, r_x, r_y, r_z] (+ gripper if is_action).
    """
    # Compute the inverse of the transformation matrix
    inverse_matrix = np.linalg.inv(transformation_matrix)
    
    # Separate position and rotation components
    position = np.array(data_vector[:3])  # Extract x', y', z' (3D vector)
    position = np.append(position, 1)     # Add homogeneous coordinate (4D vector)
    rotation = np.array(data_vector[3:6])  # Extract r_x', r_y', r_z'

    # Apply inverse transformation to position
    reverted_position = inverse_matrix @ position  # Matrix multiplication
    reverted_position = reverted_position[:3]  # Convert back to Cartesian coordinates

    # Apply the inverse rotational part of the matrix
    rotation_matrix = inverse_matrix[:3, :3]
    reverted_rotation = rotation_matrix @ rotation

    # Combine position and rotation
    reverted_vector = np.concatenate([reverted_position, reverted_rotation])


    gripper_value = data_vector[6]  # Preserve the gripper value
    reverted_vector = np.append(reverted_vector, gripper_value)

    return reverted_vector

def visualize_combined_trajectories(xyz_trajectories, yxz_trajectories):
    """
    Visualize trajectories from both datasets in a single 3D plot.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot xyz dataset trajectories
    for trajectory, gripper_states, color in xyz_trajectories:
        trajectory = np.array(trajectory)
        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        ax.plot(x, y, z, label='XYZ Dataset', color=color)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            marker = '^' if i > 0 and gripper_states[i - 1] > 0.5 else 'o'
            ax.scatter(xi, yi, zi, color=color, s=50, marker=marker)
            ax.text(xi, yi, zi, f'{i}', fontsize=8, color='black')  # Step number
    
    # Plot yxz dataset trajectories
    for trajectory, gripper_states, color in yxz_trajectories:
        trajectory = np.array(trajectory)
        x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
        ax.plot(x, y, z, label='ZYX Dataset', color=color)
        for i, (xi, yi, zi) in enumerate(zip(x, y, z)):
            marker = '^' if i > 0 and gripper_states[i - 1] > 0.5 else 'o'
            ax.scatter(xi, yi, zi, color=color, s=50, marker=marker)
            ax.text(xi, yi, zi, f'{i}', fontsize=8, color='black')  # Step number

    # Mark the start position
    ax.scatter(*START_POSITION, c='red', s=100, label='Start Position', marker='*')

    # Add labels, legend, and title
    ax.set_title('3D Trajectories for XYZ and YXZ Datasets')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()

    plt.show()

# Parameters
xyz_dir = 'data/xyz'
yxz_dir = 'data/zyx'
xyz_prefix = 'xyz_'
yxz_prefix = 'zyx_'
start_ep = int(input('Starting episode: '))
ep_num = int(input('Number of episodes: '))

# Separate colormaps for the datasets
xyz_colormap = plt.cm.Blues
yxz_colormap = plt.cm.Greens

# Process both datasets
xyz_trajectories = process_trajectory(xyz_dir, xyz_prefix, start_ep, ep_num, xyz_colormap, reorder=False)  # No reordering for xyz
yxz_trajectories = process_trajectory(yxz_dir, yxz_prefix, start_ep, ep_num, yxz_colormap, reorder=True)  # Reorder y, x, z -> x, y, z

# Visualize combined trajectories
visualize_combined_trajectories(xyz_trajectories, yxz_trajectories)

print("Processing complete!")
