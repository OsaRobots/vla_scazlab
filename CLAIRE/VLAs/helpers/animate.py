import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image

# Load your episode data
for i in range(10):
    path = f"../data/test/validate/episode_validate_{i}.npy"
    ep = np.load(path, allow_pickle=True)

    # Extract images from the episode
    images = [Image.fromarray(step['image']) for step in ep]

    # Initialize the plot
    fig, ax = plt.subplots()
    ax.axis('off')  # Turn off axes for better visualization
    img_plot = ax.imshow(images[0])

    title = ax.set_title(f"Episode: {i}", fontsize=16)

    # Animation update function
    def update(frame):
        img_plot.set_array(images[frame])
        return img_plot,

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(images), interval=100, blit=True)

    # Display the animation
    plt.show()
