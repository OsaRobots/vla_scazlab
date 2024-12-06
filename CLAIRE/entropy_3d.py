import numpy as np
import matplotlib.pyplot as plt
import csv

CLAIRE_FOLDER = "/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE"
DATA_FOLDER = f"{CLAIRE_FOLDER}/vla_run_data"
SUCCESS_TRAILS = {
    0: 1, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1, 
    8: 0, 9: 1, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
    16: 0, 17: 1, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
    24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 1, 31: 1,
    32: 1, 33: 1, 34: 1, 35: 1, 36: 1, 37: 1, 38: 1, 39: 1
}
SUCCESS_TIMESTEPS = {
    0: 23, 3: 21, 7: 23, 9: 25, 17: 27, 30: 32, 31: 41,
    32: 25, 33: 38, 34: 24, 35: 31, 36: 22, 37: 21, 38: 16, 39: 16
}
window_size = 5

# Function to calculate joint entropy
def calculate_joint_entropy(p_x, p_y, p_z):
    # Compute the joint probability distribution
    # Using the outer product to combine probabilities from each axis
    joint_prob = np.outer(np.outer(p_x, p_y).flatten(), p_z).flatten()
    
    # Normalize the probabilities to ensure they sum to 1
    joint_prob /= np.sum(joint_prob)
    
    # Exclude zero probabilities to avoid log(0)
    joint_prob = joint_prob[joint_prob > 0]
    
    # Compute and return entropy
    return -np.sum(joint_prob * np.log2(joint_prob))

# Load the episode data
success_entropy_list = []
failure_entropy_list = []
csv_data = []

for i in range (40):
    path = f"{DATA_FOLDER}/episode_test_{i}.npy"
    ep = np.load(path, allow_pickle=True)

    # Check if the episode was successful
    if SUCCESS_TRAILS[i] == 1:
        success = True
        # If successful, set limit to the timestep where the task was completed
        limit = SUCCESS_TIMESTEPS[i]
    else:
        success = False
        # Else, set limit to the last timestep
        limit = len(ep)

    # Iterate through the episode timesteps till the limit
    current_window = []
    for step in ep[:limit]:
        probs = step['probs']  # Probabilities for each axis
        p_x, p_y, p_z = probs[:3]

        # Normalize each axis probability
        p_x /= np.sum(p_x)
        p_y /= np.sum(p_y)
        p_z /= np.sum(p_z)

        # Calculate the entropy of the 3D joint probability distribution
        entropy = calculate_joint_entropy(p_x, p_y, p_z)
        current_window.append(entropy)
        if len(current_window) == window_size:
            window_avg = sum(current_window)/window_size
            success_entropy_list.append(window_avg) if success else failure_entropy_list.append(window_avg)
            # Append entropy and success label to CSV data
            csv_data.append([window_avg, 1 if success else 0])
            current_window.pop(0)

# Write data to a CSV file
csv_file_path = f"{CLAIRE_FOLDER}/entropy_3d_window{window_size}.csv"
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Entropy", "Success"])
    # Write data
    writer.writerows(csv_data)

success_entropy_list = np.array(success_entropy_list)
failure_entropy_list = np.array(failure_entropy_list)
# Get the stats on entropy
print("Success entropy avg: ", np.mean(success_entropy_list))
print("Success entropy std: ", np.std(success_entropy_list))
print("Failure entropy avg: ", np.mean(failure_entropy_list))
print("Failure entropy std: ", np.std(failure_entropy_list))

# # Plot the distribtion of entropies with a quartile boxplot
# plt.boxplot([success_entropy_list, failure_entropy_list], labels=['Success', 'Failure'])