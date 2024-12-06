import faiss
import numpy as np
import csv
import datetime

CLAIRE_FOLDER = "/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE"
DATA_FOLDER = f"{CLAIRE_FOLDER}/vla_run_data/validate"
SUCCESS_TRAILS = {
    0: 1, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 1, 
    8: 1, 9: 1}

# Function to calculate entropy
def calculate_entropy(probabilities):
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))
success_intervention = 0
failure_intervention = 0
# Stats for window size 1
mean = 0.087053
std = 0.041631
top_5 = 0.165603
between_mean = 0.086532
mean_std = mean + std
mean_std2 = mean + std*2

thresholds = {'mean + std': mean_std, 'mean + std * 2': mean_std2, '95%': top_5, 'between_mean': between_mean}

# Number of clusters
n_clusters = 5
interventions = {}
# Write data to a CSV file
csv_file_path = f"{CLAIRE_FOLDER}/semantic_entropy_3d_validate.csv"
with open(csv_file_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(["Entropy", "Episode"])

    for i in range(10):  
        path = f"{DATA_FOLDER}/episode_validate_{i}.npy"
        ep = np.load(path, allow_pickle=True)

        success = SUCCESS_TRAILS[i] == 1
        limit = len(ep)

        ep_interventions = {'mean + std': 0, 'mean + std * 2': 0, '95%': 0, 'between_mean': 0}

        for step in ep[:limit]:
            probs = step['probs']  # Probabilities for each axis
            ranges = step['range']  # Actual values for each axis

            p_x, p_y, p_z = probs[:3]
            r_x, r_y, r_z = ranges[:3]

            # Normalize probabilities
            if np.sum(p_x) > 0 and np.sum(p_y) > 0 and np.sum(p_z) > 0:
                p_x /= np.sum(p_x)
                p_y /= np.sum(p_y)
                p_z /= np.sum(p_z)
            else:
                continue

            # Combine into a single 3D array
            prob_3d = np.outer(np.outer(p_x, p_y).flatten(), p_z).flatten().astype('float32')

            start_time = datetime.datetime.now()
            # Apply Faiss k-means clustering
            kmeans = faiss.Kmeans(d=1, k=n_clusters, niter=20, verbose=False)
            kmeans.train(prob_3d.reshape(-1, 1))  # Train on the flattened probabilities

            # Assign points to clusters
            index = faiss.IndexFlatL2(1)  # L2 distance
            index.add(kmeans.centroids)
            _, cluster_labels = index.search(prob_3d.reshape(-1, 1), 1)  # Find nearest cluster
            time_for_kmeans = datetime.datetime.now() - start_time
            print(f"Time for k-means clustering: {time_for_kmeans}")

            # Calculate the clustered probabilities
            cluster_probabilities = np.zeros(n_clusters)
            for label in cluster_labels.flatten():
                cluster_probabilities[label] += 1

            # Normalize cluster probabilities
            cluster_probabilities /= np.sum(cluster_probabilities)

            # Calculate entropy for the timestep
            entropy = calculate_entropy(cluster_probabilities)
            # Write entropy and success label to CSV
            data = [[entropy, i]]
            writer.writerows(data)
            for threshold_type, threshold in thresholds.items():
                if entropy > threshold:
                    ep_interventions[threshold_type] += 1
        interventions[i] = (ep_interventions, success)

print(interventions)


