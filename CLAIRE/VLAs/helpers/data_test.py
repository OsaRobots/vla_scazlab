import argparse, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.animation as animation
from collections import Counter

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--robot", action='store_true', help="play on robot")
parser.add_argument("-g", "--gripper", action='store_true', help="check gripper sync")
parser.add_argument("-a", "--aug", action='store_true', help="image augmentation")
args = parser.parse_args()

total_steps = 0
instructions = []
ep_num = int(input('Number of episodes: '))
start_ep = int(input('Starting episode: '))

def extract_key_phrase(text):
    try:
        in_content = text.split("In:")[1].split("Out:")[0].strip()
        key_phrase = in_content.replace("What action should the robot take to ", "").replace("?", "").strip()
        return key_phrase
    except IndexError:
        return None

def count_unique_phrases(instructions):
    extracted_phrases = [extract_key_phrase(instruction) for instruction in instructions]
    # Filter out None values if the structure is not as expected in some texts
    extracted_phrases = [phrase for phrase in extracted_phrases if phrase is not None]
    
    # Count occurrences of each unique phrase
    phrase_counts = Counter(extracted_phrases)
    
    # Print each unique phrase and its count if there are duplicates
    for phrase, count in phrase_counts.items():
        print(f"'{phrase}': encountered {count} times")
    
    return phrase_counts

def process_image(image, crop_scale=0.9):
    """Applies a center crop and resizes the image."""
    temp_image = np.array(image)
    sqrt_crop_scale = np.sqrt(crop_scale)
    target_height = int(sqrt_crop_scale * temp_image.shape[0])
    target_width = int(sqrt_crop_scale * temp_image.shape[1])
    start_h = (temp_image.shape[0] - target_height) // 2
    start_w = (temp_image.shape[1] - target_width) // 2
    cropped_image = temp_image[start_h:start_h + target_height, start_w:start_w + target_width, :]
    temp_image = Image.fromarray(cropped_image)
    temp_image = temp_image.resize(image.size, Image.Resampling.BILINEAR)
    return temp_image

for j in range(start_ep, ep_num):
    ep = np.load(f'data/train/episode_{j}.npy', allow_pickle=True)

    print(f'######### EPISODE {j} #########')
    print(f'episode length is {len(ep)}')
    total_steps += len(ep)

    inst = ep[0]['language_instruction']
    instructions.append(inst)
    print(inst)

    # Set up figure and axes
    if args.aug:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # Two subplots for original and augmented
        ax[0].set_title("Original")
        ax[1].set_title("Augmented")
    else:
        fig, ax = plt.subplots()
    
    # Initialize index for animation
    index = [0]  # Use a list to make it mutable in the inner function

    # Initial image setup
    if args.aug:
        original_image = Image.fromarray(ep[index[0]]['image'])
        augmented_image = process_image(original_image)
        im_original = ax[0].imshow(original_image)
        im_augmented = ax[1].imshow(augmented_image)
    else:
        image = Image.fromarray(ep[index[0]]['image'])
        im = ax.imshow(image)
    
    # Add action text at the bottom center
    action_text = fig.text(
        0.5, 0.02, f"Action: {ep[index[0]]['action']}", 
        ha='center', va='bottom', fontsize=12, color='black'
    )
    i = 0

    def updatefig(*args_update):
        global i , grip_index, grip
        i += 1
        # i=1
        if i > len(ep)-1:
            i=0
        
        if ep[i]['action'][-1] == 0:
            action_text.set_color('green')
            if not grip:
                grip_index = i
            grip = True
        else:
            grip = False
            action_text.set_color('black')

        action_text.set_text(f"Action: {ep[i]['action']}")

        if args.aug:
            original_image = Image.fromarray(ep[i]['image'])
            augmented_image = process_image(original_image)
            im_original.set_array(original_image)
            im_augmented.set_array(augmented_image)
            return im_original, im_augmented, action_text
        else:
            image = Image.fromarray(ep[i]['image'])
            im.set_array(image)
            return im, action_text
        
    

    ani = animation.FuncAnimation(fig, updatefig, interval=20, blit=False, cache_frame_data=False)
    plt.show()

    if args.gripper:
        fig, axs = plt.subplots(1, 4)

        axs[0].imshow(Image.fromarray(ep[grip_index-1]['image']))
        axs[0].set_title(f"Action: {ep[grip_index-1]['action'][-1]}")

        axs[1].imshow(Image.fromarray(ep[grip_index]['image']))
        axs[1].set_title(f"Action: {ep[grip_index]['action'][-1]}")

        axs[2].imshow(Image.fromarray(ep[grip_index+1]['image']))
        axs[2].set_title(f"Action: {ep[grip_index+1]['action'][-1]}")

        axs[3].imshow(Image.fromarray(ep[grip_index+2]['image']))
        axs[3].set_title(f"Action: {ep[grip_index+2]['action'][-1]}")

        plt.show()

    if args.robot:
        import sys
        sys.path.append('/home/ulas/xArm-Python-SDK')
        ip = '192.168.1.205'

        from xarm.wrapper import XArmAPI

        print("Initlizing robot arm")
        arm = XArmAPI(ip)
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(state=0)
        arm.set_gripper_mode(0)
        arm.set_gripper_enable(True)
        arm.set_gripper_speed(5000)
        gripper_open = True
        print("Robot arm initilization complete")


        for step in ep:
            action = step['action']
            x   = action[0] #* 1000
            y   = action[1] #* 1000
            z   = action[2] #* 1000
            r   = action[3]
            p   = action[4]
            yaw = action[5]
            print(f"action x:{x},y:{y},z:{z},roll:{r},pitch:{p},yaw:{yaw},")
            arm.set_position(x=x,y=y,z=z,roll=r,pitch=p,yaw=yaw, relative=True, is_radian=True, wait=True)
            if action[6]>0.5 and not gripper_open:
                arm.set_gripper_position(850, wait=True)
                gripper_open = True
            elif action[6]<0.5 and gripper_open:
                arm.set_gripper_position(0, wait=True)
                gripper_open = False



        arm.disconnect()

print("########## SUMMARY ##########")
count_unique_phrases(instructions)
print(f'Total number of steps in the dataset is {total_steps}')