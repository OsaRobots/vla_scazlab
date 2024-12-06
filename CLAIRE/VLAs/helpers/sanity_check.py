import requests, argparse, time, os, sys, math, threading
import json_numpy
json_numpy.patch()
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Initialize argparse
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--aug", action='store_true', help="augument image")
args = parser.parse_args()

class RobotMind():
    def __init__(self) -> None:
        print("Sanity check")
    
    def process_image(self,image, crop_scale=0.9):
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
    
    def predict_action(self,image,instruction):
        if args.aug:
            image = self.process_image(image)
            
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


if __name__ == "__main__":
    brain = RobotMind()
    ep_num = int(input('Number of episodes: '))
    for j in range(ep_num):
        ep=np.load(f'data/train/episode_{j}.npy', allow_pickle=True)
        print(f"Episode {j}")
        inst=ep[0]['language_instruction']
        print(f"Instruction:\n{inst}")
        entropy = np.zeros([len(ep),7])
        predicted_action = np.zeros([len(ep),7])
        ground_truth_action = np.zeros([len(ep),7])
        delta_error = []
        for i in range(len(ep)):
            img = Image.fromarray(ep[i]['image'])
            
            # plt.imshow(img)
            # plt.show()
            
            a , e, p, r = brain.predict_action(img,inst)
            true_act = ep[i]['action']
            
            predicted_action[i,:]=a
            entropy[i,:]=e
            ground_truth_action[i,:]=true_act
            delta_error.append(np.mean(np.abs(a-true_act)))
        

        plt.subplot(3,3,1)
        plt.plot(range(len(ep)),predicted_action[:,0], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,0], label="True")
        plt.title('Dx')
        plt.legend()
        plt.grid()

        plt.subplot(3,3,2)
        plt.plot(range(len(ep)),predicted_action[:,1], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,1], label="True")
        plt.title('Dy')
        plt.legend()
        plt.grid()

        plt.subplot(3,3,3)
        plt.plot(range(len(ep)),predicted_action[:,2], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,2], label="True")
        plt.title('Dz')
        plt.legend()
        plt.grid()
        
        plt.subplot(3,3,4)
        plt.plot(range(len(ep)),predicted_action[:,3], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,3], label="True")
        plt.title('Drx')
        plt.legend()
        plt.grid()

        plt.subplot(3,3,5)
        plt.plot(range(len(ep)),predicted_action[:,4], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,4], label="True")
        plt.title('Dry')
        plt.legend()
        plt.grid()

        plt.subplot(3,3,6)
        plt.plot(range(len(ep)),predicted_action[:,5], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,5], label="True")
        plt.title('Drz')
        plt.legend()
        plt.grid()

        plt.subplot(3,3,7)
        plt.plot(range(len(ep)),predicted_action[:,-1], label="Predicted")
        plt.plot(range(len(ep)),ground_truth_action[:,-1], label="True")
        plt.title('Gripper')
        plt.legend()
        plt.grid()
        
        plt.subplot(3,3,8)
        plt.plot(range(len(ep)),delta_error)
        plt.title('L1 Error')
        plt.grid()

        plt.subplot(3,3,9)
        plt.plot(range(len(ep)),entropy[:,0],label='x')
        plt.plot(range(len(ep)),entropy[:,1],label='y')
        plt.plot(range(len(ep)),entropy[:,2],label='z')
        plt.plot(range(len(ep)),entropy[:,3],label='dx')
        plt.plot(range(len(ep)),entropy[:,4],label='dy')
        plt.plot(range(len(ep)),entropy[:,5],label='dz')
        plt.plot(range(len(ep)),entropy[:,6],label='g')
        plt.title('Entropy')
        plt.legend()
        plt.grid()
        plt.show()

        



     