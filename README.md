🚗 Driver Drowsiness & Yawning Detection with CARLA & MediaPipe

This project integrates real-time driver monitoring (using OpenCV & MediaPipe) with the CARLA Autonomous Driving Simulator to detect drowsiness, yawning, and unsafe head positions.
When unsafe behavior is detected, the vehicle in CARLA is stopped smoothly, and the system triggers audio alerts for driver safety.

✨ Features

✅ Eye Aspect Ratio (EAR) for detecting closed eyes & drowsiness

✅ Mouth Aspect Ratio (MAR) for yawning detection (normal & severe yawns)

✅ Head Position Tracking – detects if the driver’s face is out of the camera frame

✅ Audio Alerts using pygame (beep.mp3 + optional custom alert voice)

✅ Smooth Vehicle Stop – gradually applies brakes in CARLA instead of abrupt stopping

✅ Resume Functionality – press R to resume autopilot after a stop

✅ Camera Control – Free roam spectator view & third-person smooth follow camera

✅ Day/Night Toggle for environment simulation

✅ Weather Customization (sunlight, fog, clouds, rain, etc.)

✅ Real-time Speed Display of CARLA vehicle

🛠️ Tech Stack

Python 3.8+

CARLA Simulator 0.9.x

OpenCV

MediaPipe Face Mesh

NumPy

Pygame
 (for sound system)

SciPy
 (distance calculations)

Threading
 (for video capture & alerts)

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/your-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies
pip install opencv-python mediapipe pygame numpy scipy

4️⃣ Install & Run CARLA Simulator

Download CARLA (0.9.14 or 0.9.15 recommended) from CARLA Releases

Launch CARLA server:

./CarlaUE4.sh -quality-level=Low -opengl   # Linux/Mac
CarlaUE4.exe -quality-level=Low            # Windows

5️⃣ Run the Driver Monitoring System
python main.py

🎮 Controls
Key	Action
Q	Quit simulation
R	Resume vehicle after stop
C	Toggle Free Roam Camera
T	Toggle Day/Night

📂 Project Structure
📦 driver-drowsiness-detection
 ┣ 📜 main.py                # Main script (OpenCV + MediaPipe + CARLA)
 ┣ 📜 beep.mp3               # Beep alert sound (place in root folder)
 ┣ 📜 Yawning alert voice.mp3 # Optional custom alert sound
 ┣ 📜 README.md              # Project documentation
 ┗ 📂 assets                 # (Optional) store extra sounds/images

🚨 Detection Thresholds

EAR_THRESHOLD = 0.23 → Closed eyes detection

DROWSY_TIME_THRESHOLD = 5s → Stop car if eyes closed for 5s

YAWN_THRESHOLD = 30 → Normal yawning alert

SEVERE_YAWN_THRESHOLD = 105.5 → Severe yawning → Stop car after 5s

HEAD_OUT_OF_FRAME_THRESHOLD = 5s → Stop car if head not detected for 5s

HIGH_MAR_THRESHOLD = 100 → Trigger audio alert if MAR > 100 for 3s

📸 Demo
linkedIn url:
https://www.linkedin.com/posts/selvaraghavan-s-899367354_adas-ai-reinforcementlearning-activity-7332811386885926915-XMIN?utm_source=share&utm_medium=member_desktop&rcm=ACoAAFhUfjUBwTNTW-O_XW_E-VnsMRBnVrc7sgc

🚀 Future Improvements

🔹 Integration with real car hardware (Raspberry Pi / Jetson Nano)

🔹 Cloud-based logging of driver behavior for insurance/risk assessment

🔹 Multi-driver dataset collection for model fine-tuning

🔹 Advanced fatigue detection using heart rate / blink frequency

🤝 Contribution

Contributions are welcome!
Feel free to fork this repo and submit a pull request 🚀

📜 License

This project is licensed under the MIT License.
