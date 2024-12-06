import requests, time, os, sys, math, threading, argparse, random, copy
import json_numpy
json_numpy.patch()
import numpy as np
# import matplotlib.pyplot as plt
from entropy_detection_methods_3d import Semantic3D, Simple3D
from entropy_detection_methods_ind import Simple,Semantic
from PIL import Image
import pyrealsense2 as rs
from scipy.stats import entropy
sys.path.append('/home/ulas/xArm-Python-SDK')
from xarm.wrapper import XArmAPI

class RobotMind():
    def __init__(self) -> None:
        ip = '192.168.1.205'
        self.gripper_open = True

        print("Initlizing camera feed")
        self.pipe = rs.pipeline()
        cfg = rs.config()
        print("Pipeline is created")
        colorizer = rs.colorizer()
        profile = self.pipe.start(cfg)
        for _ in range(10):
            frameset = self.pipe.wait_for_frames()
        print("Camera feed initilization complete") 

        print("Initlizing robot arm")
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        # self.arm.set_mode(1)
        self.arm.set_mode(0)
        self.arm.set_state(state=0)
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        self.pose = self.arm.get_position(is_radian=True)[1]
        print("Robot arm initilization complete")

        # self.detector = Simple()
        # self.detector = Semantic(window=1)
        # self.detector = Semantic3D(windowsize=1)
        self.detector = Simple3D(windowsize=1, threshold='mean + std')

    def reset_arm(self,init):
        self.arm.set_gripper_position(850, wait=True)
        self.arm.set_position(x=init[0],y=init[1],z=init[2],roll=init[3],pitch=init[4],yaw=init[5], is_radian=True, wait=True)
        

    def get_image(self):
        frameset = self.pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()
        colorized_frame = np.asanyarray(color_frame.get_data(),dtype=np.uint8)
        image = Image.fromarray(colorized_frame)

        #crop image to get 720x720
        image = image.crop((280,0,1000,720))
        #resizde image to get 224x24
        image = image.resize((224,224))

        return image
    
    def get_prompt(self):
        instruction = input("Task: ")
        return instruction
    
    def predict_action(self,image,instruction):
        response = requests.post(
            "http://0.0.0.0:8000/act",
            json={"image": np.asanyarray(image, dtype=np.uint8), "instruction": instruction, "unnorm_key": "vla_dataset"}
        )
        if response.status_code == 200:
            try:
                result = response.json()  # Parse JSON response
                action = result.get("action")
                entropy = result.get("entropy")
                probs = result.get("probs")
                range = result.get("range")
            except ValueError as e:
                print("Failed to parse JSON response:", e)
        else:
            print("Error:", response.status_code)
            print("Response Text:", response.text)
        
        return action, entropy, probs, range
    
    def act(self,action):
        if not self.arm.arm.ready:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.motion_enable(enable=True)
            self.arm.set_mode(0)
            self.arm.set_state(0)

        x   = action[0]
        y   = action[1]
        z   = action[2]
        r   = action[3]
        p   = action[4]
        yaw = action[5]
        print(f"action x:{x},y:{y},z:{z},roll:{r},pitch:{p},yaw:{yaw},")

        self.arm.set_position(x=x,y=y,z=z,roll=r,pitch=p,yaw=yaw, relative=True, is_radian=True, wait=True)

        if action[6]>0.5 and not self.gripper_open:
            self.arm.set_gripper_position(850, wait=True)
            self.gripper_open = True
        elif action[6]<0.5 and self.gripper_open:
            self.arm.set_gripper_position(0, wait=True)
            self.gripper_open = False

    def check_for_help(self,p,r):
        prob = copy.deepcopy(p)
        range_val = copy.deepcopy(r)
        # return self.detector.threshold(prob)
        # return self.detector.threshold(prob,range_val)
        return self.detector.threshold_entropy(prob)
    
    def use_help(self,a,p,r):
        action = copy.deepcopy(a)
        
        # Identify top 3 actions with actual values for each axis where help is requested
        top_actions_per_axis = {}
        for dim, axis in enumerate(["X", "Y", "Z"]):
            top_indices = np.argsort(p[dim])[-3:][::-1]
            range_flipped = np.flip(r[dim])
            print(f'For {axis} axis top 3 actions are:')
            top_actions_per_axis[axis] = [(range_flipped[idx],f"Action {range_flipped[idx]:.2f} (P={p[dim][idx]:.2f})")
                for idx in top_indices
            ]
            print(top_actions_per_axis[axis])

        for key in top_actions_per_axis:
            index = int(input(f"Select action for axis {key}: "))
            if key == 'X':
                action[0]=top_actions_per_axis[key][index][0]
            elif key == 'Y':
                action[1]=top_actions_per_axis[key][index][0]
            elif key == 'Z':
                action[2]=top_actions_per_axis[key][index][0]

        return action
            

    def apply_inverse_transformation(self,data_vector, transformation_matrix):
        """
        Apply the inverse of a 4x4 transformation matrix to a data vector.
        Args:
            data_vector (np.ndarray): The transformed vector [x', y', z', r_x', r_y', r_z'] (+ gripper if is_action).
            transformation_matrix (np.ndarray): The original 4x4 transformation matrix.
            is_action (bool): Whether the input vector is an action (7 fields, with gripper as the last field).
        Returns:
            np.ndarray: Reverted vector [x, y, z, r_x, r_y, r_z] (+ gripper if is_action).
        """
        # Compute the inverse of the transformation matrix
        inverse_matrix = np.linalg.inv(transformation_matrix)
        
        # Separate position and rotation components
        position = np.array(data_vector[:3])  # Extract x', y', z' (3D vector)
        position = np.append(position, 1)     # Add homogeneous coordinate (4D vector)
        rotation = np.array(data_vector[3:6])  # Extract r_x', r_y', r_z'

        # Apply inverse transformation to position
        reverted_position = inverse_matrix @ position  # Matrix multiplication
        reverted_position = reverted_position[:3]  # Convert back to Cartesian coordinates

        # Apply the inverse rotational part of the matrix
        rotation_matrix = inverse_matrix[:3, :3]
        reverted_rotation = rotation_matrix @ rotation

        # Combine position and rotation
        reverted_vector = np.concatenate([reverted_position, reverted_rotation])


        gripper_value = data_vector[6]  # Preserve the gripper value
        reverted_vector = np.append(reverted_vector, gripper_value)

        return reverted_vector

def record_step_episode(episode, image, action, instruction, prob_dist,range):
        episode.append({
            'image': np.asarray(image, dtype=np.uint8),
            'action': action,
            'probs': prob_dist,
            'range': range,
            'language_instruction': instruction.lower(),
        })

if __name__ == "__main__":
    brain = RobotMind()
    # Initial values
    # init = [270, 50, 150, 3.14, 0.0, 0.0]
    init = [334, -37, 202, 3.14, 0.0, 0.0]

    randomize = 'n'#input("Randomize (y/n): ")
    if randomize.lower() == 'y':
        # Define ranges for x, y, z as (min, max)
        ranges = [
            (270, 360),  # Range for x
            (-70, 70),    # Range for y
            (160, 200)   # Range for z
        ]
        # Randomize x, y, z within the defined ranges
        randomized_xyz = [random.uniform(r[0], r[1]) for r in ranges]
        # Update init with the randomized values
        init[:3] = randomized_xyz

    brain.reset_arm(init)
    input("Press Enter to continue...")
    img = brain.get_image()

    # plt.imshow(img)
    # plt.show()

    inst = 'lift the coke can'#brain.get_prompt()
    
    entropies = []
    eps = []
    steps   = 0
    help_count = 0
    try:
        while True:
            # Predict
            print(f'#### STEP {steps} ####')
            a , e, p, r = brain.predict_action(img,inst)
            
            # Check to seek helps
            # help = brain.check_for_help(p,r)
            # if help:
            #     a = brain.use_help(a,p,r)
            #     help_count+=1
            
            # Recording the step
            record_step_episode(eps,img,a,inst,p,r)

            # Act
            brain.act(a)
            
            # Observe
            img=brain.get_image()

            entropies.append(e)
            steps+=1

            if steps > 40:
                break
    except KeyboardInterrupt:
        print('interrupted!')


    # path_fix = 'data/test/live/sem_ind_bottle/episode_'
    # path_fix = 'data/test/live/simp_ind_bottle/episode_'
    # path_fix = 'data/test/live/no_help_bottle/episode_'
    # path_fix = 'data/test/live/sem_3d/episode_'
    path_fix = 'data/test/live/simp_3d/episode_'
    path = input("Save episode number: ")
    np.save(path_fix+path, eps)
    print(f'Help Count {help_count}')