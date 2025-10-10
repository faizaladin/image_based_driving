import sys
import os
import glob
import time
import queue
import numpy as np
from PIL import Image
import cv2
import torch
import subprocess
import csv
import shutil

# --- Find and Add CARLA .egg to Python Path ---
try:
    CARLA_ROOT = "/home/faizaladin/Desktop/carla" # Update this path if needed
    dist_path = os.path.join(CARLA_ROOT, 'PythonAPI/carla/dist')
    egg_files = glob.glob(f'{dist_path}/carla-*-py{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    if not egg_files: raise IndexError
    sys.path.append(egg_files[0])
except IndexError:
    print("❌ Could not find a compatible CARLA .egg file.")
    sys.exit()

import carla
from model import Driving # Assumes your model class is in model.py

# --- Configuration ---
HOST = 'localhost'
PORT = 2000
MODEL_PATH = 'town4.pth'
IMAGE_WIDTH = 400
IMAGE_HEIGHT = 300
STEERING_SCALING_FACTOR = 100.0
DEVICE = torch.device('cpu')
FIXED_DELTA_SECONDS = 0.1
CARLA_EXECUTABLE = os.path.join(CARLA_ROOT, "CarlaUE4.sh")

def restart_simulator():
    """Kills any running CARLA processes and starts a new one in the background."""
    print("Killing existing Carla processes...")
    os.system("pkill -f CarlaUE4-Linux-Shipping")
    time.sleep(5)  # Allow time for system ports to be released

    print("Starting new Carla simulator instance...")
    env = os.environ.copy()
    env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
    subprocess.Popen(
        [CARLA_EXECUTABLE, "-RenderOffScreen"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT
    )
    print("Waiting 20 seconds for the simulator to initialize...")
    time.sleep(20)

def preprocess_image(carla_image):
    """
    Preprocesses a CARLA image to the format expected by the model.
    """
    array = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    array = array.reshape((carla_image.height, carla_image.width, 4))
    array = array[:, :, :3]
    img = Image.fromarray(array)
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    return img_tensor

def ensure_dirs(base_dir):
    os.makedirs(os.path.join(base_dir, "success"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "lane_violation"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "collision"), exist_ok=True)

def main():
    client = None
    world = None
    vehicle = None
    camera = None
    collision_sensor = None
    original_settings = None

    try:
        # --- Data organization setup ---
        base_dir = "/media/faizaladin/vlm_data/town4_sunny"
        ensure_dirs(base_dir)
        csv_path = os.path.join(base_dir, "run_log.csv")
        write_header = not os.path.exists(csv_path)
        if write_header:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["run_id", "violation", "violation_time", "collision_object"])

        # --- 0. Start the Simulator ---
        restart_simulator()

        # --- 1. Load Model ---
        print(f"Loading model from {MODEL_PATH} on device {DEVICE}...")
        model = Driving().to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print("✅ Model loaded successfully.")

        # --- 2. Connect to CARLA with Retries ---
        for i in range(10): 
            try:
                print(f"Attempting to connect to CARLA (Attempt {i+1}/10)...")
                client = carla.Client(HOST, PORT)
                client.set_timeout(10.0)
                print("Loading Town02...")
                world = client.load_world('Town04')
                print("✅ CARLA connection successful and Town02 loaded.")
                break
            except RuntimeError as e:
                print(f"Connection failed: {e}. Retrying in 5 seconds...")
                time.sleep(5)
        
        if not world:
            raise RuntimeError("❌ Could not connect to CARLA after multiple attempts.")

        # --- 3. Set Synchronous Mode ---
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)
        print(f"✅ CARLA set to synchronous mode with dt={FIXED_DELTA_SECONDS}s (10 FPS).")

        #world.set_weather(carla.WeatherParameters.SoftRainNoon)

        # --- 4. Spawn Vehicle and Camera ---
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = world.get_map().get_spawn_points()
        print(f"Total spawn points: {len(spawn_points)}")

        target_speed = 6.0  # m/s (about 28.8 km/h)

        # --- Loop over all spawn points ---
        for idx, spawn_point in enumerate(spawn_points):
            print(f"\n=== Starting run {idx+1} at spawn point {idx} ===")
            # Clean up previous actors
            vehicle = None
            camera = None
            collision_sensor = None
            collision_events = []
            start_frame = None
            first_lane_violation_time = None

            # Spawn vehicle and camera
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
            camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

            collision_bp = blueprint_library.find('sensor.other.collision')
            collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
            collision_events = []

            def on_collision(event):
                actor_type = event.other_actor.type_id
                impulse = event.normal_impulse
                print(f"💥 Collision with {actor_type} | Impulse: {impulse}")
                collision_events.append((event.frame, actor_type, impulse.x, impulse.y, impulse.z))

            collision_sensor.listen(on_collision)

            image_queue = queue.Queue()
            camera.listen(image_queue.put)
            world.tick()

            # --- Video writer ---
            video_width = camera_bp.get_attribute('image_size_x').as_int()
            video_height = camera_bp.get_attribute('image_size_y').as_int()
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            max_frames = int(8 / FIXED_DELTA_SECONDS)

            # Run number logic
            try:
                with open(csv_path, "r") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    if len(rows) > 1:
                        last_run = int(rows[-1][0])
                    else:
                        last_run = 0
            except FileNotFoundError:
                last_run = 0
            run_number = last_run + 1

            temp_video_filename = f"run_{run_number}.mp4"
            out = cv2.VideoWriter(temp_video_filename, fourcc, 10, (video_width, video_height))

            try:
                frame_count = 0
                while frame_count < max_frames:
                    world.tick()
                    image = image_queue.get()

                    # Start timer when car first moves (speed > 0.5 m/s)
                    if start_frame is None:
                        velocity = vehicle.get_velocity()
                        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                        if speed > 0.5:
                            start_frame = image.frame

                    image_tensor = preprocess_image(image)
                    with torch.no_grad():
                        prediction = model(image_tensor)
                    predicted_steer = prediction.item()

                    # Get current speed
                    velocity = vehicle.get_velocity()
                    speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                    speed_error = target_speed - speed
                    throttle = 0.4 + 0.1 * speed_error
                    throttle = max(0.0, min(1.0, throttle))

                    control = carla.VehicleControl(throttle=throttle, steer=predicted_steer)
                    vehicle.apply_control(control)

                    # Lane violation detection (ignore junctions)
                    if start_frame is not None:
                        vehicle_location = vehicle.get_location()
                        waypoint = world.get_map().get_waypoint(vehicle_location, project_to_road=True, lane_type=carla.LaneType.Driving)
                        if waypoint is not None and not waypoint.is_junction:
                            lane_center = waypoint.transform.location
                            lateral_dist = vehicle_location.distance(lane_center)
                            if lateral_dist > 1.5 and first_lane_violation_time is None:
                                first_lane_violation_time = (frame_count * FIXED_DELTA_SECONDS)
                        elif waypoint is None and first_lane_violation_time is None:
                            first_lane_violation_time = (frame_count * FIXED_DELTA_SECONDS)

                    raw_view = np.frombuffer(image.raw_data, dtype=np.uint8)
                    raw_view = raw_view.reshape((image.height, image.width, 4))
                    raw_view = raw_view[:, :, :3]
                    raw_view = cv2.cvtColor(raw_view, cv2.COLOR_RGB2BGR)
                    out.write(raw_view)
                    cv2.imshow("CARLA Autonomous Drive", raw_view)
                    frame_count += 1

            finally:
                out.release()
                violation_type = ""
                violation_time = ""
                violation_object = ""
                crashed = False
                crash_time = ""
                crash_object = ""
                if collision_events and start_frame is not None:
                    valid_collisions = [event for event in collision_events if event[0] >= start_frame and event[0] < start_frame + max_frames]
                    if valid_collisions:
                        first_collision = valid_collisions[0]
                        collision_frame = first_collision[0] - start_frame
                        collision_time = collision_frame * FIXED_DELTA_SECONDS
                        raw_object = first_collision[1]
                        collision_object = raw_object.split('.')[-1] if '.' in raw_object else raw_object
                        crashed = True
                        crash_time = f"{collision_time:.2f}"
                        violation_type = "collision"
                        violation_time = crash_time
                        violation_object = collision_object
                        print(f"\n❌ Car crashed! First collision at {crash_time} seconds after car started moving.")
                        print(f"   Object collided with: {collision_object}")
                    elif first_lane_violation_time is not None:
                        violation_type = "lane violation"
                        violation_time = f"{first_lane_violation_time:.2f}"
                        violation_object = ""
                        print(f"\n🚨 Lane violation at {violation_time} seconds after car started moving.")
                    else:
                        violation_type = ""
                        violation_time = ""
                        violation_object = ""
                        print("\n✅ No violation during the 8 second trajectory.")
                elif first_lane_violation_time is not None:
                    violation_type = "lane violation"
                    violation_time = f"{first_lane_violation_time:.2f}"
                    violation_object = ""
                    print(f"\n🚨 Lane violation at {violation_time} seconds after car started moving.")
                else:
                    violation_type = ""
                    violation_time = ""
                    violation_object = ""
                    print("\n✅ No violation during the 8 second trajectory.")

                # Move video to correct folder
                if violation_type == "collision":
                    dest_folder = os.path.join(base_dir, "collision")
                elif violation_type == "lane violation":
                    dest_folder = os.path.join(base_dir, "lane_violation")
                else:
                    dest_folder = os.path.join(base_dir, "success")
                dest_path = os.path.join(dest_folder, f"run_{run_number}.mp4")
                shutil.move(temp_video_filename, dest_path)

                # Log to CSV
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([run_number, violation_type, violation_time, violation_object])

                # Cleanup actors for next run
                if camera is not None:
                    camera.destroy()
                if vehicle is not None:
                    vehicle.destroy()
                if collision_sensor is not None:
                    collision_sensor.destroy()
                cv2.destroyAllWindows()

        print("\nAll runs complete.")

    finally:
        print("\nCleaning up actors and restoring settings...")
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        cv2.destroyAllWindows()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()