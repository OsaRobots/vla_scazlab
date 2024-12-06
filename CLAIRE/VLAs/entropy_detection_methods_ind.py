import numpy as np
from scipy.stats import entropy

class Simple:
    def __init__(self, threshold_x = 3.0, threshold_y = 1.7, threshold_z = 3.0, window=1):
        self.thresholds = [threshold_x, threshold_y, threshold_z]
        self.window = np.zeros((window,3))

    def threshold(self, p):
        entropies = []

        for i in range(3):
            probs = p[i]
            entropies.append(entropy(probs))

        entropy_step = np.array(entropies)

        self.window[-1,:]=entropy_step

        exceeded = np.mean(self.window) > self.thresholds
        if np.any(exceeded):
            return True
        return False
        
class Semantic:
    def __init__(self, buffer=5.0, threshold_x = 0.6, threshold_y = 0.5, threshold_z = 0.7, window=5):
        # Define buffer size
        self.buffer = buffer 
        self.thresholds = [threshold_x, threshold_y, threshold_z]
        self.window = np.zeros((window,3))

    def threshold(self, p, r):
        # Calculate binned entropies
        entropies = []

        for i in range(3):
            range_data = r[i]
            probs = p[i]
            range_data = np.flip(range_data)


            negative_bin_indices = range_data < -self.buffer
            neutral_bin_indices = (range_data >= -self.buffer) & (range_data <= self.buffer)
            positive_bin_indices = range_data > self.buffer
            
            bin_probs = [
                np.sum(probs[negative_bin_indices]),  # Negative bin
                np.sum(probs[neutral_bin_indices]),  # Neutral bin
                np.sum(probs[positive_bin_indices])  # Positive bin
            ]
            
            bin_probs = np.array(bin_probs) / np.sum(bin_probs)
            entropies.append(entropy(bin_probs))

        entropy_step = np.array(entropies)

        self.window[-1,:]=entropy_step

        exceeded = np.mean(self.window) > self.thresholds
        if np.any(exceeded):
            return True
        return False