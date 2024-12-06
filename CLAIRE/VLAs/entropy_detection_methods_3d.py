import numpy as np
import faiss
import datetime

# Stats for thresholds
window1stats = {'mean': 6.193998, 'std': 2.390957, 'top_5': 10.244, 'between_mean': 5.99}
window3stats = {'mean': 6.190764, 'std': 1.834103, 'top_5': 9.3348178, 'between_mean': 5.989}
window5stats = {'mean': 6.209817, 'std': 1.644686, 'top_5': 9.0202, 'between_mean': 6.015}
simple_threshold_stats = {1: window1stats, 3: window3stats, 5: window5stats}
# semantic_thresholds = {1: 0.086532, 3: 0.085449, 5: 0.084122}
semantic_thresholds = {1: 0.128684} # Only for one window for now

class Simple3D:
    def __init__(self, windowsize, threshold):
        self.current_window = []
        if windowsize in [1, 3, 5]:
            self.windowsize = windowsize
        else:
            print("Invalid window size. Defaulting to 5")
            self.windowsize = 5
        match threshold:
            case 'mean + std':
                self.threshold = simple_threshold_stats[windowsize]['mean'] + simple_threshold_stats[windowsize]['std']
            case 'mean + std * 2':
                self.threshold = simple_threshold_stats[windowsize]['mean'] + simple_threshold_stats[windowsize]['std'] * 2
            case '95%':
                self.threshold = simple_threshold_stats[windowsize]['top_5']
            case 'between_mean':
                self.threshold = simple_threshold_stats[windowsize]['between_mean']
            case _:
                raise ValueError("Invalid threshold method")

    def calculate_joint_entropy(self, p_x, p_y, p_z):
        # Compute the joint probability distribution
        # Using the outer product to combine probabilities from each axis
        joint_prob = np.outer(np.outer(p_x, p_y).flatten(), p_z).flatten()
        
        # Normalize the probabilities to ensure they sum to 1
        joint_prob /= np.sum(joint_prob)
        
        # Exclude zero probabilities to avoid log(0)
        joint_prob = joint_prob[joint_prob > 0]
        
        # Compute and return entropy
        return -np.sum(joint_prob * np.log2(joint_prob))
    
    """
    Returns True if the entropy of the 3D joint probability distribution entropy is above the threshold
    Returns False otherwise
    """
    def threshold_entropy(self, probs):
        p_x, p_y, p_z = probs[:3]

        # Normalize each axis probability
        p_x /= np.sum(p_x)
        p_y /= np.sum(p_y)
        p_z /= np.sum(p_z)

        # Calculate the entropy of the 3D joint probability distribution
        entropy = self.calculate_joint_entropy(p_x, p_y, p_z)
        self.current_window.append(entropy)
        if len(self.current_window) == self.windowsize:
            window_avg = sum(self.current_window)/self.windowsize
            self.current_window.pop(0)

            return window_avg > self.threshold
        
class Semantic3D:
    def __init__(self, windowsize=5, n_clusters=5):
        self.current_window = []
        if windowsize in [1, 3, 5]:
            self.windowsize = windowsize
        else:
            print("Invalid window size. Defaulting to 5")
            self.windowsize = 5
        self.threshold = semantic_thresholds[windowsize]
        self.n_clusters = n_clusters
        self.current_window = []

    def calculate_entropy(self, probabilities):
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    def kmeans_clustering(self, prob_3d):
        # Filter out values under a certain threshold
        log_prob_3d = np.log10(prob_3d + 1e-12)
        _, bin_edges = np.histogram(log_prob_3d, bins=50)
        threshold = bin_edges[25]
        filtered_prob_3d = prob_3d[log_prob_3d > threshold]

        # print(f"Number of original points: {len(prob_3d)}")
        # print(f"Number of filtered points: {len(filtered_prob_3d)}")

        # Apply Faiss k-means clustering
        kmeans = faiss.Kmeans(d=1, k=self.n_clusters, niter=20, verbose=False)
        kmeans.train(filtered_prob_3d.reshape(-1, 1))  # Train on the flattened probabilities

        # Assign points to clusters
        index = faiss.IndexFlatL2(1)  # L2 distance
        index.add(kmeans.centroids)
        _, cluster_labels = index.search(filtered_prob_3d.reshape(-1, 1), 1)  # Find nearest cluster

        return cluster_labels
    
    """
    Returns True if the entropy of the 3D joint probability distribution semantic entropy is above the threshold
    Returns False otherwise
    Assumes that the range is properly aligned to positive -> negative
    """
    def threshold_entropy(self, probs):
        p_x, p_y, p_z = probs[:3]

        # Normalize probabilities
        if np.sum(p_x) > 0 and np.sum(p_y) > 0 and np.sum(p_z) > 0:
            p_x /= np.sum(p_x)
            p_y /= np.sum(p_y)
            p_z /= np.sum(p_z)

            # Combine into a single 3D array
            prob_3d = np.outer(np.outer(p_x, p_y).flatten(), p_z).flatten().astype('float32')

            start_time = datetime.datetime.now()
            
            cluster_labels = self.kmeans_clustering(prob_3d)

            time_for_kmeans = datetime.datetime.now() - start_time
            print(f"Time for k-means clustering: {time_for_kmeans}")

            # Calculate the clustered probabilities
            cluster_probabilities = np.zeros(self.n_clusters)
            for label in cluster_labels.flatten():
                cluster_probabilities[label] += 1

            # Normalize cluster probabilities
            cluster_probabilities /= np.sum(cluster_probabilities)

            # Calculate entropy for the timestep
            entropy = self.calculate_entropy(cluster_probabilities)

            self.current_window.append(entropy)
            if len(self.current_window) == self.windowsize:
                window_avg = sum(self.current_window)/self.windowsize
                self.current_window.pop(0)

                return window_avg > self.threshold
            else:
                return False