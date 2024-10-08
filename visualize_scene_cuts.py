import cv2
import os
import pickle

# Load the scene list from the shot segmentation file
with open('scene_list.pkl', 'rb') as f:
    scene_list = pickle.load(f)

# Path to the video file
video_path = '/Users/lasyaedunuri/Documents/AML/matchCutting/ToKillAMockingBird.mp4'

# Define the output folder for saving the images
output_folder = 'scene_cuts'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# Open the video file with OpenCV
cap = cv2.VideoCapture(video_path)

# Iterate over the scene list and extract the middle frame of each scene
for i, scene in enumerate(scene_list):
    # Calculate the middle frame of the scene
    start_frame = scene[0].get_frames()
    end_frame = scene[1].get_frames()
    middle_frame = (start_frame + end_frame) // 2
    
    # Seek to the middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
    
    # Read the frame
    ret, frame = cap.read()
    
    if ret:
        # Define the output path for the image
        output_path = os.path.join(output_folder, f"scene_{i}_middle_frame.jpg")
        
        # Save the frame as an image
        cv2.imwrite(output_path, frame)
        print(f"Saved: {output_path}")
    else:
        print(f"Failed to read frame for scene {i}")

# Release the video capture object
cap.release()