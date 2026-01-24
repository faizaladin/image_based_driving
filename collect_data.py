import sys
import os
import glob
import time
import subprocess
import queue
import numpy as np
from PIL import Image

try:
    CARLA_ROOT = "/home/faizaladin/Desktop/carla"
    dist_path = os.path.join(CARLA_ROOT, 'PythonAPI/carla/dist')
    egg_files = glob.glob(f'{dist_path}/carla-*-py{sys.version_info.major}.{sys.version_info.minor}-*.egg')
    
    if not egg_files:
        raise IndexError
        
    sys.path.append(egg_files[0])

except IndexError:
    print(f"Could not find .egg file{sys.version_info.major}.{sys.version_info.minor}.")
    sys.exit()

import carla
-
CARLA_EXECUTABLE = os.path.join(CARLA_ROOT, "CarlaUE4.sh")
HOST = 'localhost'
PORT = 2000
TM_PORT = 8012
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
RESIZED_IMAGE_WIDTH = 400
RESIZED_IMAGE_HEIGHT = 300
MAPS = ["Town02"]  
FIXED_DELTA_SECONDS = 0.1 

def restart_simulator():
    """Kills any running CARLA processes and starts a new one in the background."""
    print("Killing existing Carla processes...")
    os.system("pkill -f CarlaUE4-Linux-Shipping")
    time.sleep(5)  

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

def collect_data_on_map(map_name, output_dir):
    """Connects to CARLA, runs the simulation on a given map, and collects data."""
    client = None
    world = None
    vehicle = None
    camera = None
    original_settings = None
    
    try:
        for _ in range(10): 
            try:
                client = carla.Client(HOST, PORT)
                client.set_timeout(10.0)
                world = client.load_world(map_name)
                print(f"CARLA loaded {map_name}")
                break
            except RuntimeError as e:
                print(f"Connection failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)
        
        if not world:
            raise RuntimeError("Could not connect to CARLA simulator after multiple attempts.")
        
        world.set_weather(carla.WeatherParameters.Default)

        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        tm = client.get_trafficmanager(TM_PORT)
        tm.set_synchronous_mode(True)
        tm.set_random_device_seed(42) 

        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_point = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        vehicle.set_autopilot(True, TM_PORT)
        tm.auto_lane_change(vehicle, False)
        tm.ignore_lights_percentage(vehicle, 100)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
        camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
        camera_bp.set_attribute('fov', '90')
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        os.makedirs(output_dir, exist_ok=True)
        data_labels = []
        
        image_queue = queue.Queue()
        camera.listen(image_queue.put)

        start_location = spawn_point.location
        max_frames = 8000  

        print(f"Starting data collection")
        world.tick() 

        for frame_num in range(max_frames):
            world.tick()
            
            image = image_queue.get() 
            control = vehicle.get_control() 
            
            if frame_num > 100:
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))
                
                array = array[:, :, :3]
                
                img = Image.fromarray(array)
                
                img = img.resize((RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT), Image.LANCZOS)
                
                img.save(os.path.join(output_dir, f"{image.frame}.png"))

                velocity = vehicle.get_velocity()
                speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
                location = vehicle.get_location()

                data_labels.append((image.frame, control.steer, speed, location.x, location.y, location.z))
        
    finally:
        print("Cleaning up")
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        print("Cleanup complete.")

    if data_labels:
        with open(os.path.join(output_dir, "labels.csv"), "w") as f:
            f.write("frame,steer,speed,x,y,z\n")
            for row in data_labels:
                f.write(f"{row[0]},{row[1]:.6f},{row[2]:.6f},{row[3]:.6f},{row[4]:.6f},{row[5]:.6f}\n")
        print(f"Saved {len(data_labels)} data points to '{output_dir}/labels.csv'")

def main():
    for i, map_name in enumerate(MAPS, 1):
        print(f"\n{'='*50}\nProcessing Map {i}/{len(MAPS)}: {map_name}\n{'='*50}")
        
        restart_simulator()
        
        output_dir = f"town_{i:02d}_{map_name}"
        collect_data_on_map(map_name, output_dir)
        
    print("\n Done")

if __name__ == "__main__":
    main()

