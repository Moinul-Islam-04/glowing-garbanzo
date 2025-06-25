import cv2
import json
import base64
import numpy as np
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- MediaPipe Imports ---
import mediapipe as mp

# Initialize FastAPI app
app = FastAPI()

# --- Initialize MediaPipe Pose Landmarker ---
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
print("MediaPipe Pose Landmarker model loaded successfully for side-view analysis")

# (Angle calculation function is no longer used for this logic, but can be kept)

# HTML page for testing the WebSocket
@app.get("/", response_class=HTMLResponse)
async def get():
    try:
        with open("index.html") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found.</h1>", status_code=404)

# Websocket endpoint for side-view posture analysis
@app.websocket("/ws/posture")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            
            if "," in data:
                data = data.split(',')[1]

            image_data = base64.b64decode(data)
            np_image = np.frombuffer(image_data, np.uint8)
            bgr_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=bgr_image)
            detection_result = detector.detect(mp_image)

            feedback = "Please turn to your side..."
            keypoints = {}
            
            if detection_result.pose_landmarks:
                feedback = "Posture looks good!"
                pose_landmarks = detection_result.pose_landmarks[0]
                image_height, image_width, _ = bgr_image.shape
                
                # --- NEW LOGIC: Use side-view landmark indices ---
                LEFT_EAR = 7
                LEFT_SHOULDER = 11
                LEFT_HIP = 23
                LEFT_KNEE = 25 # Adding the knee for better visualization

                # Check which side is more visible to the camera
                left_side_visibility = pose_landmarks[LEFT_SHOULDER].visibility + pose_landmarks[LEFT_HIP].visibility
                right_side_visibility = pose_landmarks[12].visibility + pose_landmarks[24].visibility

                if left_side_visibility > right_side_visibility:
                    ear_lm = pose_landmarks[LEFT_EAR]
                    shoulder_lm = pose_landmarks[LEFT_SHOULDER]
                    hip_lm = pose_landmarks[LEFT_HIP]
                    knee_lm = pose_landmarks[LEFT_KNEE]
                else: # Assume right side is visible
                    ear_lm = pose_landmarks[8] # RIGHT_EAR
                    shoulder_lm = pose_landmarks[12] # RIGHT_SHOULDER
                    hip_lm = pose_landmarks[24] # RIGHT_HIP
                    knee_lm = pose_landmarks[26] # RIGHT_KNEE
                
                required_landmarks = [ear_lm, shoulder_lm, hip_lm]

                if all(lm.visibility > 0.4 for lm in required_landmarks): # Lowered visibility threshold slightly
                    ear_coords = (int(ear_lm.x * image_width), int(ear_lm.y * image_height))
                    shoulder_coords = (int(shoulder_lm.x * image_width), int(shoulder_lm.y * image_height))
                    hip_coords = (int(hip_lm.x * image_width), int(hip_lm.y * image_height))
                    knee_coords = (int(knee_lm.x * image_width), int(knee_lm.y * image_height))
                    
                    keypoints = {
                        "EAR": ear_coords,
                        "SHOULDER": shoulder_coords,
                        "HIP": hip_coords,
                        "KNEE": knee_coords
                    }

                    # --- Posture Analysis based on X-axis alignment ---
                    # The threshold determines sensitivity. Higher value = less sensitive.
                    threshold = 40 

                    # Check for forward head posture
                    if shoulder_coords[0] - ear_coords[0] > threshold:
                        feedback = "Forward Head Posture! Align your ears with your shoulders."
                    # Check for slumped back
                    elif hip_coords[0] - shoulder_coords[0] > threshold:
                        feedback = "Slumped Back! Align your shoulders over your hips."
                    else:
                        feedback = "Great alignment!"
                else:
                    feedback = "Adjust position to make side fully visible."
            else:
                feedback = "No person detected."

            await websocket.send_text(json.dumps({"keypoints": keypoints, "feedback": feedback}))

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if websocket.client_state != "DISCONNECTED":
             await websocket.close()
        print("WebSocket connection logic finished.")