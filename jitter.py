import cv2
import mediapipe as mp
import numpy as np
import time
import carla
import pygame
import sys
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
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()

# ------------------ Vehicle Spawning (Mustang) ------------------
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find("vehicle.ford.mustang")  
spawn_points = world.get_map().get_spawn_points()

vehicle = None
for spawn_point in spawn_points:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if vehicle:
        break  

if vehicle is None:
    print("🚨 Could not spawn a Mustang! Exiting...")
    sys.exit(1)

print("✅ Mustang spawned successfully.")
vehicle.set_autopilot(True)

# ------------------ Spectator Camera ------------------
spectator = world.get_spectator()
is_free_roam = False  

def update_spectator():
    """Smoothly follow the vehicle in third-person view."""
    if not is_free_roam and vehicle.is_alive:
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

print("🎥 Smooth Spectator Camera Enabled!")

# ------------------ Initialize Webcam ------------------
video_processor = VideoProcessor()
video_processor.start()

# ------------------ Sound System ------------------
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.mp3")

# ------------------ Drowsiness Detection ------------------
drowsy_start_time = None
car_stopped = False
beep_active = False  # Flag to track if beep is playing

def smooth_stop(vehicle):
    """Gradually stop the vehicle and play beep immediately."""
    global beep_active
    print("🛑 Stopping vehicle smoothly...")
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
    print("🚨 Vehicle stopped!")

    # 🔊 *Start Continuous Beep*
    beep_active = True
    Thread(target=play_beep_loop, daemon=True).start()

def play_beep_loop():
    """Continuously play beep every BEEP_INTERVAL seconds until stopped."""
    global beep_active
    while beep_active:
        beep_sound.play()
        time.sleep(BEEP_INTERVAL)

# ------------------ Main Loop ------------------
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.65, min_tracking_confidence=0.65)

while video_processor.running:
    with video_processor.lock:
        if video_processor.frame is None:
            continue
        frame = video_processor.frame.copy()

    # Convert to RGB
    resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(resized_frame)

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

        # Debugging Messages
        print(f"Left EAR: {left_ear:.2f}, Right EAR: {right_ear:.2f}, Avg EAR: {avg_ear:.2f}")

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

    # Camera update
    if vehicle and not is_free_roam:
        update_spectator()

    # Show Frame
    cv2.imshow("Driver Monitoring", frame)

    # Key Controls
    key = cv2.waitKey(1)
    if key == ord('q'):
        beep_active = False  # Stop beep
        break
    elif key == ord('r') and car_stopped:
        print("✅ Resuming vehicle movement...")
        vehicle.set_autopilot(True)
        car_stopped = False
        beep_active = False  # *Stop Beep Immediately*
        pygame.mixer.stop()  
    elif key == ord('c'):
        is_free_roam = not is_free_roam
        print("📷 Free Roam Mode:", is_free_roam)

# Cleanup
video_processor.release()
cv2.destroyAllWindows()
if vehicle.is_alive:
    vehicle.destroy()
print("✅ Simulation Ended")
