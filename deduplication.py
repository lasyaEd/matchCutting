import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define the folder containing the middle frame images from each scene
scene_folder = 'scene_cuts'

# Load the pre-trained EfficientNet model for feature extraction
model = models.efficientnet_b7(pretrained=True)
model.eval()

# Transformation pipeline for resizing and normalizing images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to get embedding (feature vector) from an image
def get_embedding(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        embedding = model(image_tensor).numpy().flatten()
    return embedding

# Get list of all scene middle frame images
scene_images = [f for f in os.listdir(scene_folder) if f.endswith('.jpg')]

# Create a list to store embeddings and corresponding image paths
embeddings = []
image_paths = []

# Compute embeddings for each image
for image_file in scene_images:
    image_path = os.path.join(scene_folder, image_file)
    embedding = get_embedding(image_path)
    embeddings.append(embedding)
    image_paths.append(image_path)

# Convert embeddings list to a numpy array for similarity comparison
embeddings = np.array(embeddings)

# Threshold for cosine similarity - scenes with similarity above this are considered duplicates
similarity_threshold = 0.9

# Compare embeddings using cosine similarity and find near-duplicates
to_remove = set()
for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        if similarity > similarity_threshold:
            print(f"Scene {i} and Scene {j} are duplicates (Similarity: {similarity})")
            to_remove.add(j)  # Mark scene j for removal

# Remove duplicate images from the folder
for idx in sorted(to_remove, reverse=True):
    print(f"Removing duplicate scene image: {image_paths[idx]}")
    os.remove(image_paths[idx])
