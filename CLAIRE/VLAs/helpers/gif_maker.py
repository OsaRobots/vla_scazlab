import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import entropy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import entropy

# Validation dataset directory and file paths
validation_directory = "../data/test/live/sem_ind"  # Replace with your directory path
validation_files = [f"{validation_directory}/episode_{i}.npy" for i in range(0, 10)]  # Adjust filenames as needed

# Process each validation episode
for episode_idx, validation_file in enumerate(validation_files, start=1):
    # Load validation episode
    ep = np.load(validation_file, allow_pickle=True)
    
    # Extract data
    images = [Image.fromarray(step['image']) for step in ep]
    # probs = [step['probs'] for step in ep]
    # ranges = [step['range'] for step in ep]
    # actions = [step['action'] for step in ep]
    inst = [step['language_instruction'] for step in ep]
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update(frame):
        ax.clear()
        ax.imshow(images[frame])
        ax.set_title(f"Episode {episode_idx} | Step: {frame} | Inst: {inst[frame]}")
        ax.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(images), interval=200, repeat=False)
    
    # Save animation
    save_path = f"./gifs/episode_{episode_idx}.gif"
    ani.save(save_path, writer="pillow")
    print(f"Saved animation for Episode {episode_idx} to {save_path}")

plt.close('all')
