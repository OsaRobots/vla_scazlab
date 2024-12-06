import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import entropy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import entropy

# Validation dataset directory and file paths
validation_directory = "../data/test/validate"  # Replace with your directory path
validation_files = [f"{validation_directory}/episode_validate_{i}.npy" for i in range(0, 1)]  # Adjust filenames as needed

# Define buffer size
buffer = 10.0  # Adjust as needed for the "zero" range

# Thresholds for binned entropy (use statistics from training)
threshold_x = 0.6  # Replace with actual threshold for X
threshold_y = 0.4  # Replace with actual threshold for Y
threshold_z = 0.6  # Replace with actual threshold for Z
thresholds = [threshold_x, threshold_y, threshold_z]

# Function to calculate entropy for bins with a buffer
def calculate_binned_entropy(probs, range_data):
    negative_bin_indices = range_data < -buffer
    neutral_bin_indices = (range_data >= -buffer) & (range_data <= buffer)
    positive_bin_indices = range_data > buffer
    
    bin_probs = [
        np.sum(probs[negative_bin_indices]),  # Negative bin
        np.sum(probs[neutral_bin_indices]),  # Neutral bin
        np.sum(probs[positive_bin_indices])  # Positive bin
    ]
    
    bin_probs = np.array(bin_probs) / np.sum(bin_probs)
    return entropy(bin_probs)

# Process each validation episode
for episode_idx, validation_file in enumerate(validation_files, start=1):
    # Load validation episode
    ep = np.load(validation_file, allow_pickle=True)
    
    # Extract data
    images = [Image.fromarray(step['image']) for step in ep]
    probs = [step['probs'] for step in ep]
    ranges = [step['range'] for step in ep]
    actions = [step['action'] for step in ep]
    
    # Calculate binned entropies
    entropies = np.array([
        [
            calculate_binned_entropy(probs[i][0], ranges[i][0]),
            calculate_binned_entropy(probs[i][1], ranges[i][1]),
            calculate_binned_entropy(probs[i][2], ranges[i][2])
        ]
        for i in range(len(ep))
    ])
    
    # Identify help requests and top actions
    help_requests = []
    for i, (entropy_step, prob_step) in enumerate(zip(entropies, probs)):
        exceeded = entropy_step > thresholds
        if np.any(exceeded):
            responsible_axes = ["X" if exceeded[0] else "", "Y" if exceeded[1] else "", "Z" if exceeded[2] else ""]
            responsible_axes = [axis for axis in responsible_axes if axis]
            
            # Identify top 3 actions with actual values for each axis where help is requested
            top_actions_per_axis = {}
            for dim, axis in enumerate(["X", "Y", "Z"]):
                if exceeded[dim]:
                    top_indices = np.argsort(prob_step[dim])[-5:][::-1]
                    top_actions_per_axis[axis] = [
                        f"Action {ranges[i][dim][idx]:.2f} (P={prob_step[dim][idx]:.2f})"
                        for idx in top_indices
                    ]
            
            help_requests.append((i, responsible_axes, top_actions_per_axis))
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame])
        ax.set_title(f"Episode {episode_idx} | Step: {frame}")
        ax.axis("off")
        
        # Overlay help request information
        for hr in help_requests:
            if hr[0] == frame:
                responsible_axes = ", ".join(hr[1])
                top_actions_text = "\n".join(
                    [f"{axis}:\n" + "\n".join(hr[2][axis]) for axis in hr[2]]
                )
                ax.text(0.05, 0.95, f"Help Requested!\nAxes: {responsible_axes}\n\nTop Actions:\n{top_actions_text}",
                        transform=ax.transAxes, fontsize=10, color="red", va="top", bbox=dict(boxstyle="round", alpha=0.5))
    
    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=200, repeat=False)
    
    # Save animation
    save_path = f"validation_episode_{episode_idx}_help_requests.gif"
    ani.save(save_path, writer="pillow")
    print(f"Saved animation for Episode {episode_idx} to {save_path}")

plt.close('all')
