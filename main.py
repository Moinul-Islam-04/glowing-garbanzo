import cv2
import json
import base64
import numpy as np
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# (MediaPipe imports and setup are the same as before)
# --- MediaPipe Imports ---
import mediapipe as mp

app = FastAPI()

# --- Initialize MediaPipe Pose Landmarker ---
# (This section is unchanged)
model_path = 'pose_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5)

detector = PoseLandmarker.create_from_options(options)

# (Angle calculation functions are the same as before)
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def calculate_torso_angle_from_vertical(shoulder, hip):
    shoulder, hip = np.array(shoulder), np.array(hip)
    torso_vector = shoulder - hip
    vertical_vector = np.array([0, 1])
    unit_torso = torso_vector / np.linalg.norm(torso_vector)
    dot_product = np.dot(unit_torso, vertical_vector)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle_rad)

@app.get("/", response_class=HTMLResponse)
async def get():
    try:
        with open("index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)

@app.websocket("/ws/posture")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")

    # --- NEW: State variables for calibration ---
    is_calibrated = False
    calibration_data = []
    calibration_frames_needed = 60 # Ask user to hold pose for ~3 seconds (60 frames at 20fps)
    slouch_offset_threshold = 15 # How many degrees of "slouch" are allowed past calibration
    posture_threshold = 0

    try:
        while True:
            data = await websocket.receive_text()

            # Handle recalibration request from frontend
            if data == "recalibrate":
                is_calibrated = False
                calibration_data = []
                await websocket.send_text(json.dumps({"feedback": "Recalibrating... Sit up straight!", "isCalibrating": True}))
                continue

            if "," in data:
                data = data.split(',')[1]

            # (Image decoding is the same)
            image_data = base64.b64decode(data)
            bgr_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
            detection_result = detector.detect(mp_image)

            feedback = "Position yourself sideways."
            keypoints = {}
            
            if detection_result.pose_landmarks:
                # (Landmark extraction logic is the same)
                pose_landmarks = detection_result.pose_landmarks[0]
                image_height, image_width, _ = bgr_image.shape
                landmarks_indices = {'SHOULDER': (11, 12), 'HIP': (23, 24)}
                left_vis = pose_landmarks[landmarks_indices['SHOULDER'][0]].visibility
                right_vis = pose_landmarks[landmarks_indices['SHOULDER'][1]].visibility
                side_idx = 0 if left_vis > right_vis else 1
                shoulder_lm = pose_landmarks[landmarks_indices['SHOULDER'][side_idx]]
                hip_lm = pose_landmarks[landmarks_indices['HIP'][side_idx]]

                if shoulder_lm.visibility > 0.6 and hip_lm.visibility > 0.6:
                    shoulder = (shoulder_lm.x * image_width, shoulder_lm.y * image_height)
                    hip = (hip_lm.x * image_width, hip_lm.y * image_height)
                    keypoints = {"SHOULDER": [int(c) for c in shoulder], "HIP": [int(c) for c in hip]}
                    
                    current_torso_angle = calculate_torso_angle_from_vertical(shoulder, hip)

                    # --- CALIBRATION LOGIC ---
                    if not is_calibrated:
                        if len(calibration_data) < calibration_frames_needed:
                            calibration_data.append(current_torso_angle)
                            feedback = f"Calibrating... {len(calibration_data)}/{calibration_frames_needed}"
                            await websocket.send_text(json.dumps({"feedback": feedback, "isCalibrating": True, "keypoints": keypoints}))
                        else:
                            # Calculate the threshold
                            avg_good_posture_angle = np.mean(calibration_data)
                            posture_threshold = avg_good_posture_angle + slouch_offset_threshold
                            is_calibrated = True
                            feedback = "Calibration Complete!"
                            print(f"Calibration done. Threshold set to: {posture_threshold:.2f}")
                            await websocket.send_text(json.dumps({"feedback": feedback, "isCalibrating": False, "keypoints": keypoints}))
                        continue # Skip feedback logic until calibrated

                    # --- POSTURE ANALYSIS LOGIC (uses calibrated threshold) ---
                    if is_calibrated:
                        if current_torso_angle > posture_threshold:
                            feedback = "Poor Posture! Sit up straight."
                        else:
                            feedback = "Good Posture"
            else:
                feedback = "No person detected."

            await websocket.send_text(json.dumps({"keypoints": keypoints, "feedback": feedback, "isCalibrating": False}))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if websocket.client_state != "DISCONNECTED":
            await websocket.close()
        print("WebSocket connection logic finished.")