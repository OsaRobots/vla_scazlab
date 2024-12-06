import faiss
import jax.numpy as jnp
from ott.tools.k_means import k_means
import numpy as np
import csv
import datetime
import matplotlib.pyplot as plt

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
# Function to calculate entropy
# def calculate_entropy(probabilities):
#     probabilities = probabilities[probabilities > 0]
#     return -jnp.sum(probabilities * jnp.log2(probabilities))

def visualize_3d(prob_3d, p_x, p_y, p_z, cluster_labels):
    # Assume these are the dimensions of p_x, p_y, and p_z
    dim_x, dim_y, dim_z = len(p_x), len(p_y), len(p_z)

    # Reshape prob_3d and cluster_labels back to 3D
    prob_3d_reshaped = prob_3d.reshape(dim_x, dim_y, dim_z)
    cluster_labels_3d = cluster_labels.reshape(dim_x, dim_y, dim_z)

    # Create a 3D grid for plotting
    x, y, z = np.meshgrid(range(dim_x), range(dim_y), range(dim_z), indexing="ij")

    # Flatten the grid and filter points based on cluster labels
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()
    labels_flat = cluster_labels_3d.flatten()

    # Plot the 3D clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Use a scatter plot to visualize clusters
    sc = ax.scatter(x_flat, y_flat, z_flat, c=labels_flat, cmap="tab10", marker="o", alpha=0.8)

    # Add a color bar to show cluster numbers
    plt.colorbar(sc, ax=ax, label="Cluster Label")

    # Set axis labels
    ax.set_title("3D Clusters")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()

def calculate_entropy(probabilities):
    probabilities = probabilities[probabilities > 0]
    return -np.sum(probabilities * np.log2(probabilities))

# Number of clusters
n_clusters = 5

# Store entropies for each episode
all_episode_entropies = []
success_entropy_list = []
failure_entropy_list = []

def kmeans_clustering(prob_3d):
    # Filter out values under a certain threshold
    log_prob_3d = np.log10(prob_3d + 1e-12)
    _, bin_edges = np.histogram(log_prob_3d, bins=50)
    threshold = bin_edges[25]
    filtered_prob_3d = prob_3d[log_prob_3d > threshold]

    # print(f"Number of original points: {len(prob_3d)}")
    # print(f"Number of filtered points: {len(filtered_prob_3d)}")

    # Apply Faiss k-means clustering
    kmeans = faiss.Kmeans(d=1, k=n_clusters, niter=20, verbose=False)
    kmeans.train(filtered_prob_3d.reshape(-1, 1))  # Train on the flattened probabilities

    # Assign points to clusters
    index = faiss.IndexFlatL2(1)  # L2 distance
    index.add(kmeans.centroids)
    _, cluster_labels = index.search(filtered_prob_3d.reshape(-1, 1), 1)  # Find nearest cluster

    return cluster_labels

def kmeans_clustering_jax(prob_3d):
    """Performs k-means clustering using ott.tools.k_means."""
    prob_3d_jax = jnp.array(prob_3d.reshape(-1, 1))  # Convert to JAX array
    _, labels, _ = k_means(prob_3d_jax, n_clusters)
    return labels

# Write data to a CSV file
# csv_file_path = f"{CLAIRE_FOLDER}/semantic_entropy_3d_new.csv"
# with open(csv_file_path, mode='a', newline='') as file:
#     writer = csv.writer(file)
#     # Write header
#     writer.writerow(["Entropy", "Episode", "Success"])

for i in range(40):  
    path = f"{DATA_FOLDER}/episode_test_{i}.npy"
    ep = np.load(path, allow_pickle=True)

    success = SUCCESS_TRAILS[i] == 1
    limit = SUCCESS_TIMESTEPS[i] if success else len(ep)

    # Store timestep entropies for the current episode
    timestep_entropies = []
    
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
        
        cluster_labels = kmeans_clustering(prob_3d)

        # visualize_3d(prob_3d, p_x, p_y, p_z, cluster_labels)
        # cluster_labels = kmeans_clustering_jax(prob_3d)

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
        success_entropy_list.append(entropy) if success else failure_entropy_list.append(entropy)
        # Write entropy and success label to CSV
        data = [[entropy, i, 1 if success else 0]]
            # writer.writerows(data)

success_entropy_list = np.array(success_entropy_list)
failure_entropy_list = np.array(failure_entropy_list)
# Get the stats on entropy
print("Success entropy avg: ", np.mean(success_entropy_list))
print("Success entropy std: ", np.std(success_entropy_list))
print("Failure entropy avg: ", np.mean(failure_entropy_list))
print("Failure entropy std: ", np.std(failure_entropy_list))
