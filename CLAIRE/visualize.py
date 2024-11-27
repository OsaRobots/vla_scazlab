import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from PIL import Image

data_folder = "/Users/tetsu/Documents/School/Class/CPSC 473/vla_scazlab/CLAIRE/vla_run_data"

while True:
    # Load the episode data
    num = input("Ep number: ")
    path = f"{data_folder}/episode_test_{num}.npy"
    ep = np.load(path, allow_pickle=True)
    print("#### TASK ####")
    print(ep[0]['language_instruction'])

    # Function to calculate the entropy of the 3D combined probability distribution
    def calculate_entropy(prob_map):
        prob_map = prob_map.flatten()  # Flatten the 3D probability map
        prob_map = prob_map[prob_map > 0]  # Exclude zero probabilities to avoid log(0)
        return -np.sum(prob_map * np.log2(prob_map))  # Compute entropy

    # Initialize a list to store entropies and extract images
    entropies = []
    images = []

    # Process each step in the episode
    for step in ep:
        probs = step['probs']  # Probabilities for each axis
        prob_x, prob_y, prob_z = probs[:3]

        # Compute the 3D probability map as the outer product of the probabilities
        prob_map = np.outer(prob_x, prob_y).reshape(len(prob_x), len(prob_y), 1)
        prob_map = prob_map * prob_z[np.newaxis, np.newaxis, :]  # Combine with Z probabilities

        # Normalize the probability map
        prob_map /= prob_map.sum()

        # Calculate entropy for the step
        entropy = calculate_entropy(prob_map)
        entropies.append(entropy)

        # Store the image for visualization
        images.append(np.array(step['image']))  # Convert PIL Image to NumPy array

    # Matplotlib animation function
    def update(frame_idx):
        image_ax.clear()  # Clear the previous frame
        image_ax.imshow(images[frame_idx])  # Display the current image
        image_ax.axis('off')  # Remove axes for a cleaner display

        # Display step number and entropy value
        text_ax.clear()  # Clear the previous text
        text_ax.text(0.5, 0.5,
                    f"Step: {frame_idx + 1}\nEntropy: {entropies[frame_idx]:.2f} bits",
                    fontsize=16, ha='center', va='center', transform=text_ax.transAxes)
        text_ax.axis('off')  # Remove axes for the text display

    # Set up the Matplotlib figure
    fig = plt.figure(figsize=(8, 8))
    image_ax = fig.add_axes([0.05, 0.2, 0.9, 0.75])  # Main area for image
    text_ax = fig.add_axes([0.05, 0.05, 0.9, 0.1])  # Area for text

    # Create animation with slower playback (500 ms interval)
    animation = FuncAnimation(fig, update, frames=len(images), interval=500, repeat=True)

    # Show the animation
    plt.show()

    # Plot the entropy over time after the animation window is closed
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(entropies) + 1), entropies, marker='o', linestyle='-')
    plt.xlabel("Time Step")
    plt.ylabel("Entropy (bits)")
    plt.title("Entropy of 3D Combined Probability Over Time Steps")
    plt.grid()
    plt.show()