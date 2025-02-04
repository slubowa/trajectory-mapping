import os
import sys
import torch
import numpy as np
from PIL import Image as PILImage
import argparse
import carla
import atexit
import time
import random
from collections import deque

sys.path.append("/home/lubowasimon7/transfomer_project/visualnav-transformer/train")
from vint_train.models.vint.vint import ViNT

# Load ViNT Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ViNT(late_fusion=False).to(device)  # Ensure this matches what we need

# Load model weights
MODEL_WEIGHTS_PATH = "/home/lubowasimon7/transfomer_project/visualnav-transformer/deployment/model_weights/vint.pth"
try:
    checkpoint = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
except Exception as e:
    print(f"Error loading model weights: {e}")
    exit(1)

model.eval()

# Rolling buffer for past images (context queue)
context_size = 5  # Number of frames for temporal context
context_queue = deque(maxlen=context_size + 1)  # Store last `context_size` frames

# Connect to CARLA
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# Enable synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
settings.no_rendering_mode = True
world.apply_settings(settings)

blueprint_library = world.get_blueprint_library()

# Function to check if a spawn point is free
def is_spawn_point_free(world, spawn_point, radius=2.0):
    """Check if a spawn point is free within a given radius."""
    actors = world.get_actors()
    for actor in actors:
        if actor.get_transform().location.distance(spawn_point.location) < radius:
            return False  # Too close to another object
    return True  # Spawn point is free

# Load the vehicle blueprint
vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

# Get all available spawn points and shuffle them
spawn_points = world.get_map().get_spawn_points()
random.shuffle(spawn_points)  # Shuffle for random selection

# Try to find a free spawn point
vehicle = None
for spawn_point in spawn_points:
    if is_spawn_point_free(world, spawn_point):
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle:
            print(f"Vehicle spawned successfully at: {spawn_point}")
            break

# Handle the case where no valid spawn points were found
if vehicle is None:
    print("ERROR: No free spawn points available!")
    exit(1)  # Exit if no spawn points are available

# Attach a camera sensor
camera_bp = blueprint_library.find("sensor.camera.rgb")
camera_bp.set_attribute("image_size_x", "224")
camera_bp.set_attribute("image_size_y", "224")
camera_bp.set_attribute("fov", "90")

camera_transform = carla.Transform(carla.Location(x=1.5, z=2.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

print("CARLA setup complete! Running in headless mode.")

# Store past frames in context queue
def update_context(image):
    """Maintain a queue of past frames for temporal context."""
    image_tensor = preprocess_image(image)
    context_queue.append(image_tensor)

# Define goal dynamically (not using a topomap)
def get_goal_image():
    """Define goal image dynamically based on past frames."""
    if len(context_queue) < context_size:
        return None  # Not enough frames collected yet

    # For now, let's use the **oldest frame in context_queue** as the goal
    return context_queue[0]  # Goal = first image in the rolling buffer

# Preprocess Image for ViNT
def preprocess_image(image):
    """Convert CARLA image to tensor."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]  # Remove alpha channel
    image_pil = PILImage.fromarray(array).resize((224, 224))

    image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
    return image_tensor.unsqueeze(0).to(device)

# Process Image for ViNT
def process_image(image):
    """Process image from CARLA and generate waypoint prediction."""
    try:
        update_context(image)  # Store new frame in the context queue
    except Exception as e:
        print(f"[ERROR] Failed to update context queue: {e}")
        return None

    # Ensure enough frames before processing
    if len(context_queue) < context_size:
        print(f"[INFO] Skipping frame: Not enough past frames yet ({len(context_queue)}/{context_size}).")
        return None

    try:
        obs_img = torch.cat(list(context_queue)[-context_size:], dim=1)  # Use ONLY last `context_size` frames
    except Exception as e:
        print(f"[ERROR] Failed to stack past frames: {e}")
        return None

    try:
        goal_img = get_goal_image()
        if goal_img is None:
            print("[INFO] Skipping frame: No goal image available.")
            return None
    except Exception as e:
        print(f"[ERROR] Failed to get goal image: {e}")
        return None

    # Validate expected input shapes
    if obs_img.shape[1] != 15:  # 5 frames × 3 channels = 15
        print(f"[ERROR] obs_img has {obs_img.shape[1]} channels instead of 15!")
        return None

    if goal_img.shape[1] != 3:  # 1 goal frame = 3 channels
        print(f"[ERROR] goal_img has {goal_img.shape[1]} channels instead of 3!")
        return None

    try:
        obsgoal_img = torch.cat([obs_img, goal_img], dim=1)  # Expected shape: [1, 18, 224, 224]
    except Exception as e:
        print(f"[ERROR] Failed to concatenate obs_img and goal_img: {e}")
        return None

    print(f"[DEBUG] obs_img shape: {obs_img.shape}, goal_img shape: {goal_img.shape}, obsgoal_img shape: {obsgoal_img.shape}")

    try:
        with torch.no_grad():
            distance_pred, waypoints_pred = model(obsgoal_img, goal_img)
    except Exception as e:
        print(f"[ERROR] Model inference failed: {e}")
        return None

    return waypoints_pred[0].cpu().numpy()
    
# Camera callback function
def camera_callback(image):
    waypoint = process_image(image)
    if waypoint is not None:
        print(f"Predicted Waypoint: {waypoint}")

# Attach callback to camera
camera.listen(lambda image: camera_callback(image))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy ViNT in CARLA (Headless Mode)")
    parser.add_argument("--model", "-m", default="vint", type=str, help="Model name")
    args = parser.parse_args()

    print("ViNT is running inside CARLA in headless mode.")

    try:
        while settings.synchronous_mode:
            world.tick()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Shutting down...")
