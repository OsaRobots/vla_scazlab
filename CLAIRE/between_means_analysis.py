import pandas as pd
import numpy as np

# Load the CSV file
file_path = "/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE/entropy_data/semantic_entropy_3d_new.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Function to calculate sliding window averages
def calculate_sliding_window_averages(data, window_size):
    results = []
    grouped = data.groupby("Episode")
    
    for episode, group in grouped:
        entropies = group["Entropy"].values
        success = group["Success"].values  # Keep Success values for each step
        # Compute sliding window averages
        averages = [
            np.average(entropies[i : i + window_size]) 
            for i in range(window_size-1, len(entropies))
        ]
        # Add to results
        for i, avg in enumerate(averages):
            results.append({
                "Episode": episode,
                "Timestep": i + 1,
                "Window_Avg_Entropy": avg,
                "Success": success[i]  # Keep the corresponding Success label
            })
    return pd.DataFrame(results)

# Compute sliding window averages for window sizes 3 and 5
window_3_df = calculate_sliding_window_averages(data, window_size=3)
window_5_df = calculate_sliding_window_averages(data, window_size=5)

# Save to new CSV files
# window_3_df.to_csv("semantic_entropy_3d_window3.csv", index=False)
# window_5_df.to_csv("semantic_entropy_3d_window5.csv", index=False)


# Get stats for the sliding window averages
window_3_stats = window_3_df.groupby("Success")["Window_Avg_Entropy"].describe()
window_5_stats = window_5_df.groupby("Success")["Window_Avg_Entropy"].describe()

# Get the mean for success and failure
success_mean_3 = window_3_stats.loc[1, "mean"]
failure_mean_3 = window_3_stats.loc[0, "mean"]
success_mean_5 = window_5_stats.loc[1, "mean"]
failure_mean_5 = window_5_stats.loc[0, "mean"]

between_means_3 = (success_mean_3 + failure_mean_3) / 2
between_means_5 = (success_mean_5 + failure_mean_5) / 2

threshold3 = between_means_3
threshold5 = between_means_5
print(f"Threshold for window size 3: {threshold3}")
print(f"Threshold for window size 5: {threshold5}")

validate_df = pd.read_csv("/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE/entropy_data/semantic_entropy_3d_validate.csv")

# Group the data by 'Episode'
grouped_validate = validate_df.groupby("Episode")

# For each episode, go through the timesteps, and count how many time the entropy is above the threshold
for episode, group in grouped_validate:
    print(f"Episode {episode}:")
    entropies = group["Entropy"].values  # Extract the entropies for the episode
    window_size = 3
    threshold = threshold3
    # Compute sliding window averages
    averages = [
        np.mean(entropies[i : i + window_size]) 
        for i in range(window_size-1, len(entropies))
    ]
    # Count the number of times the window average exceeds the threshold
    count = sum(avg > threshold for avg in averages)
    print(f"Exceeded threshold {count} times")
