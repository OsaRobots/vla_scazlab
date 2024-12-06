import numpy as np
from scipy.stats import entropy
import pandas as pd

# Directory containing the 40 episodes
directory = "../data/test"  # Replace with your directory path
file_paths = [f"{directory}/episode_test_{i}.npy" for i in range(0, 40)]  # Adjust filenames as needed

# Define buffer size
b = float(input("buffer (mm): "))
buffer = b#30.0  # Adjust as needed for the "zero" range

# Function to calculate entropy for bins with a buffer
def calculate_binned_entropy(probs, range_data):
    range_data = np.flip(range_data)
    # Define bins with a buffer area around zero
    negative_bin_indices = range_data < -buffer
    neutral_bin_indices = (range_data >= -buffer) & (range_data <= buffer)
    positive_bin_indices = range_data > buffer
    
    # Sum probabilities for each bin
    bin_probs = [
        np.sum(probs[negative_bin_indices]),  # Negative bin
        np.sum(probs[neutral_bin_indices]),  # Neutral bin
        np.sum(probs[positive_bin_indices])  # Positive bin
    ]
    
    # Normalize probabilities for entropy calculation
    bin_probs = np.array(bin_probs) / np.sum(bin_probs)
    
    # Calculate entropy
    return entropy(bin_probs)

# Initialize lists to store entropies for all episodes
all_entropies_x = []
all_entropies_y = []
all_entropies_z = []

# Process each episode
for file_path in file_paths:
    # Load the episode
    ep = np.load(file_path, allow_pickle=True)
    
    # Initialize lists for this episode
    episode_entropies_x = []
    episode_entropies_y = []
    episode_entropies_z = []
    
    # Process each step in the episode
    for step in ep:
        probs = step['probs']
        range_data = step['range']
        
        # Calculate binned entropies for x, y, z
        episode_entropies_x.append(calculate_binned_entropy(probs[0], range_data[0]))
        episode_entropies_y.append(calculate_binned_entropy(probs[1], range_data[1]))
        episode_entropies_z.append(calculate_binned_entropy(probs[2], range_data[2]))
    
    # Store entropies for the episode
    all_entropies_x.extend(episode_entropies_x)
    all_entropies_y.extend(episode_entropies_y)
    all_entropies_z.extend(episode_entropies_z)

# Convert to numpy arrays for analysis
all_entropies_x = np.array(all_entropies_x)
all_entropies_y = np.array(all_entropies_y)
all_entropies_z = np.array(all_entropies_z)

# Calculate statistics
stats = {
    "Axis": ["X", "Y", "Z"],
    "Mean": [np.mean(all_entropies_x), np.mean(all_entropies_y), np.mean(all_entropies_z)],
    "Std Dev": [np.std(all_entropies_x), np.std(all_entropies_y), np.std(all_entropies_z)],
    "Min": [np.min(all_entropies_x), np.min(all_entropies_y), np.min(all_entropies_z)],
    "Max": [np.max(all_entropies_x), np.max(all_entropies_y), np.max(all_entropies_z)],
    "95th Percentile": [
        np.percentile(all_entropies_x, 95),
        np.percentile(all_entropies_y, 95),
        np.percentile(all_entropies_z, 95),
    ],
}

# Create a DataFrame for better visualization
stats_df = pd.DataFrame(stats)

# Display statistics
print(stats_df)
