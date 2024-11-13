import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Define the folder where flow images are saved
flow_folder = 'flow_images'

# Function to flatten an image for comparison
def flatten_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read as grayscale for simplicity
    return image.flatten()  # Flatten the image to a 1D array

# Get list of flow images
flow_images = sorted([f for f in os.listdir(flow_folder) if f.endswith('.jpg')])

# Flatten the flow images for comparison
flattened_flows = []
for image_file in flow_images:
    image_path = os.path.join(flow_folder, image_file)
    flattened_image = flatten_image(image_path)
    flattened_flows.append(flattened_image)

# Convert the list to a numpy array for easy comparison
flattened_flows = np.array(flattened_flows)

# Threshold for similarity (higher means more similar)
similarity_threshold = 0.85

# Compare optical flow images using cosine similarity
for i in range(len(flattened_flows)):
    for j in range(i + 1, len(flattened_flows)):
        similarity = cosine_similarity([flattened_flows[i]], [flattened_flows[j]])[0][0]
        if similarity > similarity_threshold:
            print(f"Flow image {i} and Flow image {j} are similar (Similarity: {similarity})")
