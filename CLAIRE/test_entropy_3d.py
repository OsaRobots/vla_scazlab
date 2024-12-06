import numpy as np
import matplotlib.pyplot as plt
import csv

CLAIRE_FOLDER = "/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE"
DATA_FOLDER = f"{CLAIRE_FOLDER}/vla_run_data/validate"
SUCCESS_TRAILS = {
    0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 
    8: 1, 9: 1}

success_intervention = 0
failure_intervention = 0
# Stats for window size 1
# mean = 6.193998
# std = 2.390957
# top_5 = 10.244
# between_mean = 5.99

# Stats for window size 3
# mean = 6.190764
# std = 1.834103
# top_5 = 9.3348178
# between_mean = 5.989

# Stats for window size 5
mean = 6.209817
std = 1.644686
top_5 = 9.0202
between_mean = 6.015

threshold = between_mean
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

# Load the validation data
interventions = {}
for i in range (10):
    path = f"{DATA_FOLDER}/episode_validate_{i}.npy"
    ep = np.load(path, allow_pickle=True)

    # # Check if the episode was successful
    limit = len(ep)
    success = SUCCESS_TRAILS[i] == 1

   # Iterate through the episode timesteps till the limit
    ep_intervention = 0
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
            current_entropy = sum(current_window)/window_size
            current_window.pop(0)

            if current_entropy > threshold:
                ep_intervention += 1
                if success:
                    success_intervention += 1
                else:
                    failure_intervention += 1
    interventions[i] = (ep_intervention, success)

print(interventions)
print(f"Success Intervention: {success_intervention}")
print(f"Failure Intervention: {failure_intervention}")