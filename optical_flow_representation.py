import cv2
import numpy as np
import os


# Define the folder containing the middle frame images from each scene
scene_folder = 'scene_cuts'

# Create a folder for saving flow images
flow_folder = 'flow_images'
if not os.path.exists(flow_folder):
    os.makedirs(flow_folder)
    print(f"Created folder: {flow_folder}")

# Function to compute dense optical flow between two consecutive frames
def compute_optical_flow(image1_path, image2_path):
    # Read the images in grayscale
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Ensure the images are the same size (resize if necessary)
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Compute optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

# Function to visualize and save optical flow
def visualize_optical_flow(flow, save_path):
    # Compute magnitude and angle of flow vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Normalize magnitude to a 0-255 range
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert to 8-bit image
    mag = np.uint8(mag)
    
    # Create a color representation (optional)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2  # Angle in degrees
    hsv[..., 1] = 255  # Saturation
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # Magnitude as value
    
    # Convert HSV to RGB
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Save the flow visualization as a JPG file
    cv2.imwrite(save_path, rgb_flow)

# Get list of scene images (middle frames)
scene_images = sorted([f for f in os.listdir(scene_folder) if f.endswith('.jpg')])

# Compute and save optical flow images between consecutive frames
for i in range(len(scene_images) - 1):
    image1_path = os.path.join(scene_folder, scene_images[i])
    image2_path = os.path.join(scene_folder, scene_images[i + 1])
    
    # Compute optical flow
    flow = compute_optical_flow(image1_path, image2_path)
    
    # Save the optical flow as a JPG in the flow_images folder
    flow_image_path = os.path.join(flow_folder, f"flow_{i}.jpg")
    visualize_optical_flow(flow, save_path=flow_image_path)
    print(f"Saved: {flow_image_path}")