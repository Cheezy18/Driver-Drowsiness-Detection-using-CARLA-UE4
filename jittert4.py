import cv2
import mediapipe as mp
import numpy as np
import time
import carla
import pygame
import sys
import random
import os
from scipy.spatial import distance
from threading import Thread, Lock

# ------------------ Configuration Constants ------------------
EAR_THRESHOLD = 0.25  
DROWSY_TIME_THRESHOLD = 5  
BEEP_INTERVAL = 2  

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ------------------ Video Processor Class ------------------
class VideoProcessor:
    def __init__(self):
        self.frame = None
        self.lock = Lock()
        self.running = True

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Webcam not detected! Exiting...")
            sys.exit(1)

    def start(self):
        Thread(target=self._capture, daemon=True).start()

    def _capture(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def release(self):
        self.running = False
        self.cap.release()

# ------------------ EAR Calculation ------------------
def eye_aspect_ratio(eye):
    try:
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    except:
        return 0.0

# ------------------ CARLA Connection ------------------
print("🔄 Connecting to CARLA...")
client = carla.Client("localhost", 2000)
client.set_timeout(20.0)  # Increased timeout for map loading

# List available maps and attempt to load Town04
try:
    available_maps = client.get_available_maps()
    print("🗺️ Available maps:")
    for i, map_name in enumerate(available_maps):
        print(f"  {i}: {map_name}")
    
    # Find Town04 in available maps
    town04_map = None
    for map_name in available_maps:
        if "Town04" in map_name:
            town04_map = map_name
            break
    
    if town04_map:
        print(f"🌆 Loading {town04_map}...")
        client.load_world(town04_map)
    else:
        print("⚠️ Town04 not found in available maps. Using current map.")
except Exception as e:
    print(f"❌ Error listing/loading maps: {e}")
    print("⚠️ Continuing with current map...")

world = client.get_world()
current_map = world.get_map()
print(f"🌍 Current map: {current_map.name}")

# ------------------ Vehicle Spawning (Mustang) ------------------
print("🔍 Finding vehicle blueprints...")
blueprint_library = world.get_blueprint_library()
try:
    vehicle_bp = blueprint_library.find("vehicle.ford.mustang")
    print("✅ Found Mustang blueprint")
except:
    # Fallback to any car if Mustang not available
    print("⚠️ Mustang not found in blueprint library. Using alternative vehicle.")
    vehicle_bp = blueprint_library.filter("vehicle.audi")[0]

print("🔍 Finding spawn points...")
spawn_points = world.get_map().get_spawn_points()
if not spawn_points:
    print("❌ No spawn points found! Exiting...")
    sys.exit(1)

print(f"✅ Found {len(spawn_points)} spawn points")
random.shuffle(spawn_points)  # Randomize spawn points

# Try to spawn the player vehicle (Mustang or alternative)
print("🚗 Attempting to spawn player vehicle...")
vehicle = None
for spawn_point in spawn_points[:10]:  # Try first 10 spawn points
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        print(f"✅ {vehicle_bp.id} spawned successfully.")
        break

if vehicle is None:
    print("🚨 Could not spawn player vehicle! Exiting...")
    sys.exit(1)

# Enable autopilot for player vehicle
try:
    vehicle.set_autopilot(True)
    print("✅ Autopilot enabled for player vehicle")
except Exception as e:
    print(f"❌ Error enabling autopilot: {e}")
    print("⚠️ Vehicle will not move automatically")

# ------------------ Spectator Camera ------------------
spectator = world.get_spectator()
is_free_roam = False  

def update_spectator():
    """Smoothly follow the vehicle in third-person view."""
    if not is_free_roam and vehicle and vehicle.is_alive:
        try:
            transform = vehicle.get_transform()
            target_location = transform.location + carla.Location(-6 * transform.get_forward_vector().x,
                                                                 -6 * transform.get_forward_vector().y,
                                                                 3)  

            current_transform = spectator.get_transform()
            smooth_location = carla.Location(
                0.1 * target_location.x + 0.9 * current_transform.location.x,
                0.1 * target_location.y + 0.9 * current_transform.location.y,
                0.1 * target_location.z + 0.9 * current_transform.location.z
            )

            spectator.set_transform(carla.Transform(smooth_location, transform.rotation))
        except Exception as e:
            pass  # Silently handle any errors during camera update

print("🎥 Smooth Spectator Camera Enabled!")

# ------------------ Initialize Webcam ------------------
print("📹 Initializing webcam...")
video_processor = VideoProcessor()
video_processor.start()
print("✅ Webcam started")

# ------------------ Sound System ------------------
print("🔊 Initializing sound system...")
pygame.mixer.init()
# Check if beep.mp3 exists, if not create a fallback plan
beep_sound = None
try:
    if os.path.exists("beep.mp3"):
        beep_sound = pygame.mixer.Sound("beep.mp3")
        print("✅ Loaded beep sound")
    else:
        print("⚠️ beep.mp3 not found. Using console alerts instead.")
except Exception as e:
    print(f"❌ Error initializing sound: {e}")

# ------------------ Drowsiness Detection ------------------
drowsy_start_time = None
car_stopped = False
beep_active = False  # Flag to track if beep is playing

def smooth_stop(vehicle):
    """Gradually stop the vehicle and play beep immediately."""
    global beep_active
    print("🛑 Stopping vehicle smoothly...")
    
    try:
        vehicle.set_autopilot(False)
        control = carla.VehicleControl()
        
        for brake in np.linspace(0.1, 1.0, 5):
            control.throttle = 0.0
            control.brake = brake
            vehicle.apply_control(control)
            time.sleep(0.3)

        control.brake = 1.0
        control.hand_brake = True
        vehicle.apply_control(control)
    except Exception as e:
        print(f"❌ Error applying vehicle control: {e}")
    
    print("🚨 Vehicle stopped!")

    # 🔊 *Start Continuous Beep*
    beep_active = True
    Thread(target=play_beep_loop, daemon=True).start()

def play_beep_loop():
    """Continuously play beep every BEEP_INTERVAL seconds until stopped."""
    global beep_active
    while beep_active:
        if beep_sound:
            beep_sound.play()
        else:
            print("🔊 BEEP! (Driver drowsy alert)")
        time.sleep(BEEP_INTERVAL)

# ------------------ Weather Setting ------------------
try:
    weather = world.get_weather()
    weather.sun_altitude_angle = 70  # Daytime
    weather.cloudiness = 20
    weather.precipitation = 0
    weather.precipitation_deposits = 0
    weather.wind_intensity = 10
    weather.fog_density = 5
    world.set_weather(weather)
    print("☀️ Weather set to sunny day with light clouds")
except Exception as e:
    print(f"❌ Error setting weather: {e}")

# ------------------ Main Loop ------------------
print("🎭 Initializing MediaPipe Face Mesh...")
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.65, min_tracking_confidence=0.65)
print("✅ Face Mesh initialized")

print("🎮 Controls:")
print("   - Q: Quit simulation")
print("   - R: Resume driving when stopped")
print("   - C: Toggle free roam camera mode")
print("   - T: Toggle day/night")
print("👁️ Starting drowsiness detection loop...")

try:
    while video_processor.running:
        # Update spectator camera first
        if vehicle and not is_free_roam:
            update_spectator()
            
        # Process webcam frame
        with video_processor.lock:
            if video_processor.frame is None:
                time.sleep(0.01)  # Small delay to prevent CPU hogging
                continue
            frame = video_processor.frame.copy()

        # Convert to RGB
        resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(resized_frame)

        avg_ear = 0.0  # Default value
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Get eye coordinates
            h, w, _ = frame.shape
            left_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

            # Compute EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            # Drowsiness Detection
            if avg_ear < EAR_THRESHOLD:
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                
                elapsed_time = time.time() - drowsy_start_time
                cv2.putText(frame, f"DROWSY! Stopping in {max(0, DROWSY_TIME_THRESHOLD - elapsed_time):.1f}s",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                if elapsed_time >= DROWSY_TIME_THRESHOLD and not car_stopped:
                    smooth_stop(vehicle)
                    car_stopped = True

            else:
                drowsy_start_time = None
                cv2.putText(frame, "Alert", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display EAR value
        if results.multi_face_landmarks:
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show vehicle speed
        if vehicle:
            try:
                velocity = vehicle.get_velocity()
                speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
                cv2.putText(frame, f"Speed: {speed:.1f} km/h", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            except:
                cv2.putText(frame, "Speed: N/A", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Show Frame
        cv2.imshow("Driver Monitoring", frame)

        # Key Controls
        key = cv2.waitKey(1)
        if key == ord('q'):
            beep_active = False  # Stop beep
            break
        elif key == ord('r') and car_stopped:
            print("✅ Resuming vehicle movement...")
            try:
                vehicle.set_autopilot(True)
                car_stopped = False
                beep_active = False  # *Stop Beep Immediately*
                if beep_sound:
                    pygame.mixer.stop()  
            except Exception as e:
                print(f"❌ Error resuming vehicle: {e}")
        elif key == ord('c'):
            is_free_roam = not is_free_roam
            print("📷 Free Roam Mode:", is_free_roam)
        elif key == ord('t'):
            # Toggle between day and night
            try:
                day_mode = not day_mode if 'day_mode' in locals() else False
                weather = world.get_weather()
                if day_mode:
                    weather.sun_altitude_angle = 70  # Day
                    print("☀️ Switching to daytime")
                else:
                    weather.sun_altitude_angle = -30  # Night
                    print("🌙 Switching to nighttime")
                world.set_weather(weather)
            except Exception as e:
                print(f"❌ Error changing time of day: {e}")

except KeyboardInterrupt:
    print("👋 Program interrupted by user")
except Exception as e:
    print(f"❌ Error in main loop: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Cleanup
    print("🧹 Cleaning up simulation...")
    video_processor.release()
    cv2.destroyAllWindows()
    
    try:
        pygame.mixer.quit()
    except:
        pass

    # Destroy player vehicle
    print("🚗 Removing vehicle...")
    if vehicle and vehicle.is_alive:
        vehicle.destroy()
        
    print("✅ Simulation Ended")
