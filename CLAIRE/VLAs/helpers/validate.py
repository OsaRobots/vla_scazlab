import numpy as np
import pandas as pd
from scipy.stats import entropy

# Validation dataset directory and file paths
validation_directory = "../data/test/validate"  # Replace with your directory path
validation_files = [f"{validation_directory}/episode_validate_{i}.npy" for i in range(0, 10)]  # Adjust filenames as needed

# Predefined thresholds for x, y, z (from previous analysis)
threshold_x = 3.6  # Replace with actual threshold
threshold_y = 2.9  # Replace with actual threshold
threshold_z = 3.6  # Replace with actual threshold

# Function to count help-seeking instances
def count_help_requests(entropies, thresholds, chunk_size):
    counts = 0
    num_chunks = len(entropies) - chunk_size + 1  # Number of chunks to analyze
    for i in range(num_chunks):
        # Get chunk of entropies for x, y, z
        chunk = entropies[i:i+chunk_size]
        # Average entropy over the chunk
        avg_entropy = np.mean(chunk, axis=0)
        exceeded = avg_entropy > thresholds
        
        if np.any(exceeded):
            counts += 1
    return counts

# Process validation files
chunk_sizes = [1, 3, 5]  # Step chunking options
thresholds = [threshold_x, threshold_y, threshold_z]

results = {chunk_size: [] for chunk_size in chunk_sizes}

for episode_number, file_path in enumerate(validation_files, start=0):
    # Load validation episode
    ep = np.load(file_path, allow_pickle=True)
    
    # Extract entropies for x, y, z
    entropies = np.array([[entropy(step['probs'][dim]) for dim in range(3)] for step in ep])
    
    # Count help requests for each chunk size and condition
    for chunk_size in chunk_sizes:
            count = count_help_requests(entropies, thresholds, chunk_size, )
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