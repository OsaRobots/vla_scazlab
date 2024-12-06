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

# Pygame initialization
pygame.init()
pygame.joystick.init()

class Controller:
    def __init__(self, arm, joystick):
        self.arm = arm
        self.joystick = joystick

        self.multiplier = 100.0
        self.gripper_state = 'open'
        self.gripper = {'open': 850, 'close': 0}
        self.gripper_state_change = False

        self.right_clicked = False
        self.reset = False
        
        self.sensitivity = 0.3

        self.last_set_velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


        _, self.last_position = self.arm.get_position(is_radian=True)
        self.arm.set_cartesian_velo_continuous(True)

        signal.signal(signal.SIGINT, self.sigint_handler)

    def sigint_handler(self, sig, frame):
        print('You pressed Ctrl+C!')
        self.arm.disconnect()
        sys.exit(0)

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

if __name__ == '__main__':
    arm = XArmAPI(XARM_IP)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    controller = Controller(arm, joystick)
    controller.run_forever()
