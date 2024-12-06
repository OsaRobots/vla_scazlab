import numpy as np
import pandas as pd
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Validation dataset directory and file paths
validation_directory = "../data/test/validate"  # Replace with your directory path
validation_files = [f"{validation_directory}/episode_validate_{i}.npy" for i in range(0, 10)]  # Adjust filenames as needed

# Define buffer size
buffer = 1.0  # Adjust as needed for the "zero" range

# Thresholds for binned entropy (use statistics from training)
threshold_x = 0.7  # Replace with actual threshold for X
threshold_y = 0.7  # Replace with actual threshold for Y
threshold_z = 0.9  # Replace with actual threshold for Z
thresholds = [threshold_x, threshold_y, threshold_z]

# Function to calculate entropy for bins with a buffer
def calculate_binned_entropy(probs, range_data):
    range_data = np.flip(range_data)
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

# Function to count help requests
def count_help_requests(entropies, thresholds, chunk_size):
    counts = 0
    num_chunks = len(entropies) - chunk_size + 1
    for i in range(num_chunks):
        # Average entropy over the chunk
        chunk_avg = np.mean(entropies[i:i+chunk_size], axis=0)
        exceeded = chunk_avg > thresholds
    
        if np.any(exceeded):
            counts += 1
    return counts

# Process validation files
chunk_sizes = [1, 3, 5]
results = {chunk_size: [] for chunk_size in chunk_sizes}

for episode_number, file_path in enumerate(validation_files, start=0):
    # Load validation episode
    ep = np.load(file_path, allow_pickle=True)
    
    # Calculate binned entropies for the entire episode
    entropies = np.array([
        [
            calculate_binned_entropy(step['probs'][0], step['range'][0]),
            calculate_binned_entropy(step['probs'][1], step['range'][1]),
            calculate_binned_entropy(step['probs'][2], step['range'][2])
        ]
        for step in ep
    ])
    
    # Count help requests for each chunk size and condition
    for chunk_size in chunk_sizes:
        count = count_help_requests(entropies, thresholds, chunk_size)
        results[chunk_size].append({
            "Episode": episode_number,
            "Help Requests": count
        })

# Display results
for chunk_size, result in results.items():
    print(f"  Chunk Size: {chunk_size}")
    chunk_df = pd.DataFrame(result)
    print(chunk_df.to_string(index=False))  # Display without index for cleaner output
    print("-" * 40)
