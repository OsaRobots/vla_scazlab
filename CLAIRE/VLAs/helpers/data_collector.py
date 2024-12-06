import sys, time, json, signal, argparse
import numpy as np
sys.path.append('/home/ulas/xArm-Python-SDK')
from xarm.wrapper import XArmAPI
import pyrealsense2 as rs
from PIL import Image 
import pygame
import cv2  # Added for webcam support

import matplotlib.pyplot as plt

XARM_IP = '192.168.1.205'

# Initialize argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--meter", action='store_true', help="save in meters")
parser.add_argument("-c", "--camera", type=str, required=True, choices=["realsense", "webcam"], 
                    help="Specify the camera to use: 'realsense' or 'webcam'")
parser.add_argument('-p','--play',action='store_true', help='move only do not record data')
args = parser.parse_args()

# Pygame initialization
pygame.init()
pygame.joystick.init()

# Define data collector class for Xbox controller
class DataCollector:
    def __init__(self, arm, joystick, camera_type):
        self.arm = arm
        self.joystick = joystick
        self.camera_type = camera_type
        
        if self.camera_type == "realsense":
            self.pipe = rs.pipeline()
            cfg = rs.config()
            self.pipe.start(cfg)
            for _ in range(10):  # Adjust sync and exposure
                frameset = self.pipe.wait_for_frames()
        elif self.camera_type == "webcam":
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not self.cap.isOpened():
                raise Exception("Error: Cannot access the webcam.")
        
        self.multiplier = 100.0
        self.gripper_state = 'open'
        self.gripper = {'open': 850, 'close': 0}
        self.gripper_state_change = False

        self.right_clicked = False
        self.reset = False
        
        self.sensitivity = 0.3

        self.episode = []
        
        if args.play:
            self.episode_count = -100
            self.instruction = 'record'
        else:
            self.episode_count = int(input('Episode number: '))
            self.instruction = input("Instruction: ")
        
        self.observation = {'image': None, 'state': None}
        self.last_set_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        self.mm_to_m = np.ones(6)
        if args.meter:
            self.mm_to_m[0:3] = 1000.0

        _, self.last_position = self.arm.get_position(is_radian=True)
        self.arm.set_cartesian_velo_continuous(True)

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.arm.disconnect()
        if self.camera_type == "webcam":
            self.cap.release()
        elif self.camera_type == "realsense":
            self.pipe.stop()
        sys.exit(0)

    def get_image(self):
        if self.camera_type == "realsense":
            frameset = self.pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            if not color_frame:
                return None
            colorized_frame = np.asanyarray(color_frame.get_data(), dtype=np.uint8)
            image = Image.fromarray(colorized_frame)
        elif self.camera_type == "webcam":
            self.cap.grab()  # Clear the buffers
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                return None
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Crop and resize the image
        image = image.crop((280, 0, 1000, 720)).resize((224, 224))
        return image

    def record_step_episode(self, image, state, action, instruction):
        self.episode.append({
            'image': np.asarray(image, dtype=np.uint8),
            'state': state,
            'action': action,
            'language_instruction': f"In: What action should the robot take to {instruction.lower()}?\nOut:",
        })

    def save_episode(self, path):
        print(f'episode_{self.episode_count} is {len(self.episode)} steps long')
        np.save(path, self.episode)
        self.episode_count += 1

    def run_forever(self):
        while True:
            loop_start_time = time.time()

            if not self.arm.connected:
                self.arm.connect()
            elif not self.arm.arm.ready:
                self.arm.clean_error()
                self.arm.clean_warn()
                self.arm.motion_enable(enable=True)
                self.arm.set_mode(5)
                self.arm.set_state(0)

            # Xbox controller handling
            if self.joystick is not None:
                if self.arm.mode != 5:
                    self.arm.set_mode(5)
                    self.arm.set_state(0)
                    time.sleep(1)  # Avoid buffering
                    continue

                # Reset for new episode
                if self.joystick.get_button(0) == 1:  # 'A' button
                    self.reset_episode()
                    self.reset = True

                # Save episode
                if self.joystick.get_button(5) == 1:  # Right bumper button
                    self.save_episode(f'data/train/episode_{self.episode_count}.npy')
                    self.episode = []
                    self.reset = False

                pygame.event.pump()  # Process input events

                # Joysticks
                j0 = -self.joystick.get_axis(0)
                j1 = self.joystick.get_axis(1)
                j2 = self.joystick.get_axis(2)
                j3 = -self.joystick.get_axis(3)
                j4 = -self.joystick.get_axis(4)
                j5 = -self.joystick.get_axis(5)

                h0 = self.joystick.get_hat(0)[0]

                if abs(j0) < self.sensitivity: j0 = 0
                if abs(j1) < self.sensitivity: j1 = 0
                if abs(j3) < self.sensitivity: j3 = 0
                if abs(j4) < self.sensitivity: j4 = 0

                # Axis values for translation (right joystick)
                vy = j3 * self.multiplier
                vx = j4 * self.multiplier
                vz = j5 * self.multiplier + j2 * self.multiplier

                # Axis values for rotation (left joystick)
                vr = j0 * self.multiplier * 0.5
                vp = j1 * self.multiplier * 0.5
                vya = h0 * self.multiplier * 0.5

                velocity = [vx, vy, vz, vr, vp, vya]

                # Send velocity command if there's any movement
                if any(abs(v) > 0.1 for v in velocity) or any(abs(v) > 0 for v in self.last_set_velocity):
                    self.arm.vc_set_cartesian_velocity(velocity, duration=0.2)
                    self.last_set_velocity = velocity

                # Gripper control
                if self.joystick.get_button(4) == 1:  # Left bumper button
                    self.toggle_gripper()

            # Collect delta position and image every 0.2 seconds (5 Hz)
            if self.reset:
                self.save_step()

            elapsed_loop_time = time.time() - loop_start_time
            sleep_time = max(0, 0.2 - elapsed_loop_time)  # Ensure positive sleep time
            time.sleep(sleep_time)

    def toggle_gripper(self):
        self.gripper_state_change = True
        if self.gripper_state == 'open':
            self.arm.set_gripper_position(self.gripper['close'], wait=True)
            self.gripper_state = 'close'
        else:
            self.arm.set_gripper_position(self.gripper['open'], wait=True)
            self.gripper_state = 'open'

    def reset_episode(self):
        print("Resetting...")
        _, self.last_position = self.arm.get_position(is_radian=True)
        img = self.get_image()
        # Display the image
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title("Press any key to close")

        # Function to handle key press
        def on_key(event):
            print(f"Key pressed: {event.key}. Closing...")
            plt.close(fig)  # Close the figure window

        # Connect the key press event to the figure
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show(block=True)  # Blocking show to wait for the key press

        state = np.zeros(8)
        state[:6] = np.array(self.last_position) / self.mm_to_m
        state[7] = 1.0 if self.gripper_state == 'open' else 0.0
        self.observation['image'] = img
        self.observation['state'] = state
        self.gripper_state_change = False

    def save_step(self):
        _, curpos = self.arm.get_position(is_radian=True)
        img = self.get_image()
        state = np.zeros(8)
        state[:6] = np.array(self.last_position) / self.mm_to_m
        action_xyzrpy = np.array(curpos) - np.array(self.last_position)
        if sum(abs(action_xyzrpy[:3])) > 10.0 or self.gripper_state_change or sum(abs(action_xyzrpy[3:])) > 0.1:
            print(f"Saving step with threshold {sum(abs(action_xyzrpy))}")
            action = np.zeros(7)
            action[:6] = action_xyzrpy / self.mm_to_m
            action[6] = 1.0 if self.gripper_state == 'open' else 0.0
            state[7] = action[6]
            self.record_step_episode(self.observation['image'], self.observation['state'], action, self.instruction)
            self.last_position = curpos
            self.observation['image'] = img
            self.observation['state'] = state
            self.gripper_state_change = False
        else:
            print("No action, not saving")

if __name__ == '__main__':
    arm = XArmAPI(XARM_IP)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    controller = DataCollector(arm, joystick, args.camera)
    controller.run_forever()
