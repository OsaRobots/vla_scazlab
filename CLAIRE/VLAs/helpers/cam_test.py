# import pyrealsense2 as rs
# import numpy as np
# import cv2

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()

# # Enable the color stream
# config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# # Start streaming
# pipeline.start(config)

# try:
#     while True:
#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         color_frame = frames.get_color_frame()
        
#         if not color_frame:
#             continue

#         # Convert color frame to numpy array
#         color_image = np.asanyarray(color_frame.get_data())

#         # Get the dimensions of the original image
#         height, width, _ = color_image.shape

#         # Calculate cropping area for 720x720
#         new_dim = 720
#         start_x = (width // 2) - (new_dim // 2)
#         start_y = (height // 2) - (new_dim // 2)

#         # Crop the image to 720x720 from the center
#         cropped_image = color_image[start_y:start_y + new_dim, start_x:start_x + new_dim]

#         # Resize the cropped image to 224x224
#         # resized_image = cv2.resize(cropped_image, (224, 224))

#         # Display the resulting resized image as a video
#         cv2.imshow('RealSense Cropped and Resized Image', cropped_image)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

# finally:
#     # Stop streaming
#     pipeline.stop()
#     cv2.destroyAllWindows()

import pyrealsense2 as rs
import numpy as np
import cv2
import argparse

def main(camera_type):
    if camera_type == "realsense":
        # Configure RealSense depth and color streams
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable the color stream
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start RealSense streaming
        pipeline.start(config)

        print("Using RealSense camera...")
    elif camera_type == "webcam":
        # Open the default webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print("Error: Cannot access the webcam.")
            return

        print("Using webcam...")
    else:
        print("Invalid camera type. Please choose 'realsense' or 'webcam'.")
        return

    try:
        while True:
            if camera_type == "realsense":
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue

                # Convert color frame to numpy array
                color_image = np.asanyarray(color_frame.get_data())
            elif camera_type == "webcam":
                # Capture frame from webcam
                ret, color_image = cap.read()
                if not ret:
                    print("Error: Failed to capture image from webcam.")
                    break

            # Get the dimensions of the original image
            height, width, _ = color_image.shape

            # Calculate cropping area for 720x720
            new_dim = 720
            start_x = (width // 2) - (new_dim // 2)
            start_y = (height // 2) - (new_dim // 2)

            # Crop the image to 720x720 from the center
            cropped_image = color_image[start_y:start_y + new_dim, start_x:start_x + new_dim]

            # Resize the cropped image to 224x224
            # resized_image = cv2.resize(cropped_image, (224, 224))

            # Display the resulting resized image as a video
            cv2.imshow('Cropped and Resized Image', cropped_image)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if camera_type == "realsense":
            # Stop RealSense streaming
            pipeline.stop()
        elif camera_type == "webcam":
            # Release the webcam
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch between RealSense and webcam.")
    parser.add_argument("--camera", type=str, required=True, 
                        help="Specify the camera to use: 'realsense' or 'webcam'")
    args = parser.parse_args()
    main(args.camera)
