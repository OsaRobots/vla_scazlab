import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Camera position
CAMERA_POSITION = np.array([100, -300, 400])  # (x, y, z) in mm

def visualize_cumulative_trajectory(cumulative_trajectory):
    """
    Visualize the cumulative trajectory in 3D space with the camera position.
    """
    # Extract Cartesian components
    x, y, z = cumulative_trajectory[:, 0], cumulative_trajectory[:, 1], cumulative_trajectory[:, 2]

    # Create a 3D plot for cumulative trajectory
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot for cumulative trajectory
    ax.scatter(x, y, z, c=np.linspace(0, 1, len(x)), cmap='viridis', s=10)
    
    # Mark the camera position
    ax.scatter(*CAMERA_POSITION, color='red', s=100, label='Camera Position')

    # Add labels, legend, and title
    ax.set_title('Cumulative 3D Trajectory Volume Coverage')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()

    plt.show()

def visualize_trajectory_with_gripper(states, episode_num, cumulative_trajectory):
    """
    Visualize the trajectory in 3D space using states and mark the camera position.
    """
    # Extract Cartesian components
    trajectory = states[:, :3]  # Assuming states include x, y, z as the first three columns
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # Append trajectory to cumulative
    cumulative_trajectory.append(trajectory)

    # Gripper action (7th value): 1 = open, 0 = closed
    gripper_actions = states[:, 6]

    # Use rotational components for color mapping
    roll, pitch, yaw = states[:, 3], states[:, 4], states[:, 5]
    color_map = np.linalg.norm([roll, pitch, yaw], axis=0)  # Combine rotational magnitudes

    # 3D Scatter plot with gripper actions
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(x)):
        # Change marker based on gripper state
        marker = '^' if gripper_actions[i] > 0.5 else 'o'  # Open: '^', Closed: 'o'
        ax.scatter(x[i], y[i], z[i], c=[color_map[i]], cmap='viridis', s=50, marker=marker)
    
    # Mark the camera position
    ax.scatter(*CAMERA_POSITION, color='red', s=100, label='Camera Position')

    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='Rotation Magnitude')

    # Add labels, legend, and title
    ax.set_title(f'3D Trajectory with Gripper Actions for Episode {episode_num}')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.legend()

    plt.show()

# Function to process and save episodes
def process_episodes(start_ep, ep_num, data_dir, show_cumulative_only=False):
    """
    Process and visualize episodes, optionally only showing cumulative coverage.
    """
    cumulative_trajectory = []

    for j in range(start_ep, ep_num):
        # Load episode data
        ep_path = os.path.join(data_dir, f'episode_{j}.npy')
        if not os.path.exists(ep_path):
            print(f"Episode file {ep_path} not found!")
            continue

        ep = np.load(ep_path, allow_pickle=True)

        print(f'Processing Episode {j}')
        states = np.array([step['state'] for step in ep])  # Extract states

        # If not showing cumulative only, visualize individual trajectory
        if not show_cumulative_only:
            visualize_trajectory_with_gripper(states, j, cumulative_trajectory)

        # Update cumulative trajectory
        cumulative_trajectory.append(states[:, :3])

    # Combine cumulative trajectory into a single numpy array
    cumulative_trajectory = np.vstack(cumulative_trajectory)
    visualize_cumulative_trajectory(cumulative_trajectory)

# Parameters
data_dir = 'data/train'  # Directory containing the dataset
start_ep = int(input('Starting episode: '))
ep_num = int(input('Number of episodes: '))
show_cumulative_only = input('Show only cumulative trajectory? (yes/no): ').strip().lower() == 'yes'

# Process episodes
process_episodes(start_ep, ep_num, data_dir, show_cumulative_only)

print("Processing complete!")
