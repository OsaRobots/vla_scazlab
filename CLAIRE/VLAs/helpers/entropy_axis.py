import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Directory containing the 30 .npy files
directory = "../data/test"  # Replace with your directory path
file_paths = [f"{directory}/episode_test_{i}.npy" for i in range(0, 39)]  # Adjust filenames as needed

# Initialize lists to aggregate entropy values
entropy_x_all = []
entropy_y_all = []
entropy_z_all = []

# Process each episode
for file_path in file_paths:
    ep = np.load(file_path, allow_pickle=True)
    for step in ep:
        probs = step['probs']
        
        # Compute entropy for x, y, z
        entropy_x_all.append(entropy(probs[0]))
        entropy_y_all.append(entropy(probs[1]))
        entropy_z_all.append(entropy(probs[2]))

# Analyze distributions
def analyze_entropy(entropy_values, axis_name):
    print(f"Analysis for {axis_name}:")
    print(f"  Mean: {np.mean(entropy_values):.4f}")
    print(f"  Std: {np.std(entropy_values):.4f}")
    print(f"  95th Percentile: {np.percentile(entropy_values, 95):.4f}")
    print(f"  Max: {np.max(entropy_values):.4f}\n")

    # Plot histogram
    plt.figure(figsize=(8, 4))
    plt.hist(entropy_values, bins=30, alpha=0.7)
    plt.title(f"Entropy Distribution for {axis_name}")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.grid()
    plt.show()

# Analyze and plot for each axis
analyze_entropy(entropy_x_all, "X")
analyze_entropy(entropy_y_all, "Y")
analyze_entropy(entropy_z_all, "Z")
