import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Directory containing training episodes
training_directory = "../data/test"   # Replace with your training directory path
training_files = [f"{training_directory}/episode_test_{i}.npy" for i in range(0, 39)]  # Adjust filenames if needed

# Function to clean an episode by marking cut points
def clean_and_save_episode(file_path):
    # Load the episode
    ep = np.load(file_path, allow_pickle=True)
    
    # Extract images
    images = [Image.fromarray(step['image']) for step in ep]
    language_instruction = ep[0]['language_instruction']  # Assuming instruction is the same for the episode
    cut_point = None  # Store the index to cut the episode

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(8, 6))
    current_step = [0]  # Use a list to modify the value inside the event handler

    # Event handler for key press
    def on_key(event):
        nonlocal cut_point
        if event.key == "right":  # Move to the next step
            current_step[0] = min(current_step[0] + 1, len(images) - 1)
        elif event.key == "left":  # Move to the previous step
            current_step[0] = max(current_step[0] - 1, 0)
        elif event.key == "m":  # Mark the current step as the cut point
            cut_point = current_step[0]
            print(f"Marked step {cut_point} as the cut point.")
        elif event.key == "q":  # Quit and save
            plt.close(fig)
        update_plot()  # Update the displayed image

    # Function to update the plot
    def update_plot():
        ax.clear()
        ax.imshow(images[current_step[0]])
        ax.set_title(f"Step: {current_step[0]} | Instruction: {language_instruction}\nPress 'm' to mark, 'q' to quit")
        ax.axis("off")
        fig.canvas.draw()

    # Connect the event handler
    fig.canvas.mpl_connect("key_press_event", on_key)
    update_plot()
    plt.show()

    # Cut the episode based on the marked point
    if cut_point is not None:
        cut_end = cut_point + 1  # Include the marked point
        trimmed_episode = ep[:cut_end]
        print(f"Cutting episode from start to step {cut_point} (inclusive).")

        # Save the trimmed episode back to the same location
        np.save(file_path, trimmed_episode)
        print(f"Trimmed episode saved to {file_path}.")
    else:
        print(f"No cut point marked for {file_path}. Episode left unchanged.")

# Process each episode
for episode_idx, file_path in enumerate(training_files, start=1):
    print(f"Processing Episode {episode_idx}...")
    clean_and_save_episode(file_path)
    print(f"Finished processing Episode {episode_idx}.")

print("All episodes processed and saved.")