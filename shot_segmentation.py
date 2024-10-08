from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import pickle

# Path to the video file
video_path = '/Users/lasyaedunuri/Documents/AML/matchCutting/ToKillAMockingBird.mp4'

# Create video and scene managers
print("Initializing video manager...")
video_manager = VideoManager([video_path])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=7.0))

# Downscale for faster processing
print("Setting downscale factor for faster processing...")
video_manager.set_downscale_factor(2)

# Start the video manager
print("Starting video manager...")
video_manager.start()

# Perform shot segmentation
print("Detecting scenes...")
scene_manager.detect_scenes(video_manager)

# Get the list of detected scenes (shots)
scene_list = scene_manager.get_scene_list()

if not scene_list:
    print("No scenes detected.")
else:
    print(f"Number of scenes detected: {len(scene_list)}")

# Print each scene's start and end time
for i, scene in enumerate(scene_list):
    print(f"Scene {i}: Start {scene[0].get_timecode()}, End {scene[1].get_timecode()}")


# Save the scene list to a file using Pickle
with open('scene_list.pkl', 'wb') as f:
    pickle.dump(scene_list, f)

print("Scene list saved to 'scene_list.pkl'")
