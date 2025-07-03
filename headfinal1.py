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
EAR_THRESHOLD = 0.23 # Updated from 0.20 to 0.25 as requested
DROWSY_TIME_THRESHOLD = 5  # 5 seconds before stopping
BEEP_INTERVAL = 2  
# New configuration constants for head pose and yawning
HEAD_OUT_OF_FRAME_THRESHOLD = 5  # Seconds before alerting when head is out of frame
YAWN_THRESHOLD = 30  # Basic threshold from paper
SEVERE_YAWN_THRESHOLD = 105.5  # New threshold for severe yawning that will stop the car
MAR_CONSECUTIVE_FRAMES = 20  # Frames needed for yawn detection
YAWN_TIME_THRESHOLD = 5  # Seconds of severe yawning before stopping the car
# New threshold for high MAR alert
HIGH_MAR_THRESHOLD = 100.0  # Threshold for high MAR value
HIGH_MAR_TIME_THRESHOLD = 3  # Seconds of high MAR before audio alert

# ------------------ MediaPipe Setup ------------------
mp_face_mesh = mp.solutions.face_mesh
# Eye landmarks - using standard MediaPipe indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # MediaPipe's landmarks for left eye
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # MediaPipe's landmarks for right eye

# Mouth landmarks for yawn detection - using more precise inner lip landmarks
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# Face boundary landmarks for head position tracking
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

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

# ------------------ Improved Mouth Aspect Ratio (MAR) for Yawn Detection ------------------
def mouth_aspect_ratio(landmarks, frame_width, frame_height):
    """
    Calculate MAR according to the paper's formula:
    MAR = ((d1+d2+d3)) / (2*d4)
    
    Where:
    - d1, d2, d3 are vertical distances
    - d4 is horizontal distance
    """
    try:
        # Get inner mouth points coordinates
        mouth_points = []
        for i in INNER_LIPS:
            x = int(landmarks.landmark[i].x * frame_width)
            y = int(landmarks.landmark[i].y * frame_height)
            mouth_points.append((x, y))
            
        # Calculate vertical opening (average of multiple points for robustness)
        # Find top and bottom mouth points
        top_points = mouth_points[:len(mouth_points)//2]
        bottom_points = mouth_points[len(mouth_points)//2:]
        
        # Calculate multiple vertical distances
        vertical_distances = []
        for i in range(min(len(top_points), len(bottom_points))):
            vertical_distances.append(distance.euclidean(top_points[i], bottom_points[i]))
        
        # Average vertical distance
        vertical_dist = sum(vertical_distances) / len(vertical_distances)
        
        # Calculate horizontal width (using leftmost and rightmost points)
        # Find leftmost and rightmost points
        x_values = [p[0] for p in mouth_points]
        leftmost_idx = x_values.index(min(x_values))
        rightmost_idx = x_values.index(max(x_values))
        horizontal_dist = distance.euclidean(mouth_points[leftmost_idx], mouth_points[rightmost_idx])
        
        # Calculate MAR according to the paper's formula
        mar = (vertical_dist * 3) / (2.0 * horizontal_dist) if horizontal_dist > 0 else 0
        
        # Scale to match the paper's threshold range (0-30)
        mar = mar * 100
        
        return mar
    except Exception as e:
        print(f"Error in MAR calculation: {e}")
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

# Load custom alert sound
alert_sound = None
try:
    if os.path.exists("Yawning alert voice.mp3"):  # Replace with your audio file name
        alert_sound = pygame.mixer.Sound("Yawning alert voice.mp3")
        print("✅ Loaded alert message sound")
    else:
        print("Yawning alert voice.mp3 not found. Using console alerts instead.")
except Exception as e:
    print(f"❌ Error loading alert sound: {e}")

# ------------------ Drowsiness Detection ------------------
drowsy_start_time = None
head_out_start_time = None  # Time when head started being out of frame
yawn_start_time = None      # Time when yawning started
severe_yawn_start_time = None  # Time when severe yawning started
high_mar_start_time = None  # Time when high MAR started
mar_alert_played = False    # Flag to track if MAR alert sound played
yawn_counter = 0            # Counter for consecutive frames with yawn detected
car_stopped = False
beep_active = False         # Flag to track if beep is playing
last_mar_values = []        # Store recent MAR values for smoothing
# Added for more robust drowsiness detection
ear_history = []           # Store recent EAR values for smoothing
EAR_HISTORY_LENGTH = 10    # Number of frames to keep in history for smoothing

def smooth_stop(vehicle, reason="drowsiness"):
    """Gradually stop the vehicle and play beep immediately."""
    global beep_active, car_stopped
    print(f"🛑 Stopping vehicle smoothly due to {reason}...")
    
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
    car_stopped = True

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
            print("🔊 BEEP! (Driver alert)")
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

day_mode = True  # Initialize day_mode variable

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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        h, w, _ = frame.shape
        
        # Get face dimensions for tracking
        face_detected = False
        face_box = None
        head_position_alert = False
        yawning_alert = False
        severe_yawning_alert = False  # New flag for severe yawning
        
        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0]
            
            # Get eye coordinates
            left_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

            # Get face outline for head position tracking
            face_outline = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in FACE_OUTLINE])
            
            # Calculate face bounding box
            x_min = np.min(face_outline[:, 0])
            y_min = np.min(face_outline[:, 1])
            x_max = np.max(face_outline[:, 0])
            y_max = np.max(face_outline[:, 1])
            face_box = (x_min, y_min, x_max, y_max)
            
            # Draw face bounding box
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            
            # Check if face is near edge of frame (head position tracking)
            frame_margin = 20  # pixels from edge to consider as "out of frame"
            if (x_min < frame_margin or x_max > w - frame_margin or 
                y_min < frame_margin or y_max > h - frame_margin):
                # Head position alert - face too close to edge of frame
                if head_out_start_time is None:
                    head_out_start_time = time.time()
                
                elapsed_time = time.time() - head_out_start_time
                if elapsed_time >= HEAD_OUT_OF_FRAME_THRESHOLD:
                    head_position_alert = True
                
                cv2.putText(frame, f"Head Position Alert! {max(0, HEAD_OUT_OF_FRAME_THRESHOLD - elapsed_time):.1f}s", 
                            (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                head_out_start_time = None
            
            # Compute EAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2
            
            # Add to EAR history for smoothing
            ear_history.append(avg_ear)
            if len(ear_history) > EAR_HISTORY_LENGTH:
                ear_history.pop(0)
            
            # Calculate smoothed EAR to reduce false positives
            smoothed_ear = sum(ear_history) / len(ear_history) if ear_history else avg_ear
            
            # Calculate MAR for yawn detection using the paper's method
            mar = mouth_aspect_ratio(landmarks, w, h)
            
            # Add to MAR history for smoothing
            last_mar_values.append(mar)
            if len(last_mar_values) > 5:  # Keep last 5 values for smoothing
                last_mar_values.pop(0)
            
            # Calculate smoothed MAR to reduce false positives
            smoothed_mar = sum(last_mar_values) / len(last_mar_values)
            
            # Debug - Draw eyes
            for eye in [left_eye, right_eye]:
                cv2.polylines(frame, [np.array(eye)], True, (0, 255, 255), 1)
            
            # Draw inner lip landmarks
            inner_lips_points = np.array([(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in INNER_LIPS])
            cv2.polylines(frame, [inner_lips_points], True, (255, 0, 255), 1)
            
            # Detect high MAR values
            if smoothed_mar > HIGH_MAR_THRESHOLD:
                if high_mar_start_time is None:
                    high_mar_start_time = time.time()
                
                elapsed_time = time.time() - high_mar_start_time
                if elapsed_time >= HIGH_MAR_TIME_THRESHOLD and not mar_alert_played:
                    # Play audio alert
                    if alert_sound:
                        alert_sound.play()
                    else:
                        print("🔊 ALERT: Yawning detected!")
                    
                    mar_alert_played = True
                    
                # Display high MAR warning
                cv2.putText(frame, f"HIGH MAR DETECTED! Alert in {max(0, HIGH_MAR_TIME_THRESHOLD - elapsed_time):.1f}s", 
                            (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                high_mar_start_time = None
                mar_alert_played = False
            
            # Yawn detection with updated thresholds
            if smoothed_mar > YAWN_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= MAR_CONSECUTIVE_FRAMES:
                    if yawn_start_time is None:
                        yawn_start_time = time.time()
                    
                    # Check for severe yawning (MAR >= 105.5)
                    if smoothed_mar >= SEVERE_YAWN_THRESHOLD:
                        if severe_yawn_start_time is None:
                            severe_yawn_start_time = time.time()
                        
                        elapsed_time = time.time() - severe_yawn_start_time
                        if elapsed_time >= YAWN_TIME_THRESHOLD:
                            severe_yawning_alert = True
                        
                        # Use a different color (bright red) for severe yawn alert
                        cv2.putText(frame, f"SEVERE YAWN ALERT! Car stopping in {max(0, YAWN_TIME_THRESHOLD - elapsed_time):.1f}s", 
                                   (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        severe_yawn_start_time = None
                        # Regular yawn alert
                        cv2.putText(frame, "YAWN ALERT!", 
                                   (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
            else:
                yawn_counter = max(0, yawn_counter - 1)  # Gradually reduce counter instead of resetting
                if yawn_counter == 0:
                    yawn_start_time = None
                    severe_yawn_start_time = None
            
            # Display MAR value
            cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Enhanced Drowsiness Detection based on EAR
            if smoothed_ear < EAR_THRESHOLD:  # Using updated threshold value (0.25)
                if drowsy_start_time is None:
                    drowsy_start_time = time.time()
                
                elapsed_time = time.time() - drowsy_start_time
                # Make EAR warning more visible
                cv2.putText(frame, f"DROWSY! Stopping in {max(0, DROWSY_TIME_THRESHOLD - elapsed_time):.1f}s",
                            (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Debug information for EAR detection
                cv2.putText(frame, f"EAR: {smoothed_ear:.2f} < {EAR_THRESHOLD} (ALERT!)", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if elapsed_time >= DROWSY_TIME_THRESHOLD and not car_stopped:
                    smooth_stop(vehicle, "closed eyes")
            else:
                drowsy_start_time = None
                cv2.putText(frame, "Eyes: Alert", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        else:
            # No face detected
            if head_out_start_time is None:
                head_out_start_time = time.time()
            
            elapsed_time = time.time() - head_out_start_time
            cv2.putText(frame, f"No face detected! Alert in {max(0, HEAD_OUT_OF_FRAME_THRESHOLD - elapsed_time):.1f}s", 
                        (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if elapsed_time >= HEAD_OUT_OF_FRAME_THRESHOLD:
                head_position_alert = True

        # Stop vehicle if any drowsiness indicators are active
        if (head_position_alert or severe_yawning_alert) and not car_stopped:
            stop_reason = "head position" if head_position_alert else "severe yawning"
            smooth_stop(vehicle, stop_reason)

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
                day_mode = not day_mode
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
