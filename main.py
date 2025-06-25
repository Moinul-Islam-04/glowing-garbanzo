import cv2
import json
import base64
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from transformers import pipeline
import math

# Initialize FastAPI app
app = FastAPI()
# Load the image captioning model
posture = pipeline("posture-detection", model="facebook/detr-resnet-50-panoptic")
print("Model loaded successfully")

# angle calculation function
def calculate_angle(a, b, c):
    ab = math.dist(a, b)
    bc = math.dist(b, c)
    ac = math.dist(a, c)
    # Using the cosine rule to calculate the angle
    angle = math.acos((ab**2 + ac**2 - bc**2) / (2 * ab * ac))
    return math.degrees(angle)

# HTML page for testing the WebSocket
@app.get("/", response_class=HTMLResponse)
async def get():
    with open("index.html") as f:
        return HTMLResponse(content=f.read())
# Websocket endpoint
@app.websocket("/ws/posture")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            print("Received data:", data)
            # Decode the base64 image
            image_data = base64.b64decode(data)
            # Convert the byte data to a numpy array
            np_image = np.frombuffer(image_data, np.uint8)
            # Decode the image
            image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Perform posture detection
            results = posture(image)
            print("Posture detection results:", results)

            #filter for most confident person detection
            if results:
                person = max(results, key=lambda x: x['score'])
                if person['score'] < 0.9:
                    await websocket.send_text("No person detected with high confidence")
                    continue
                # Extract keypoints
                keypoints = {kp['keypoints']: kp['position'] for kp in person['keypoints']}
                print("Extracted keypoints:", keypoints)
                # posture logic
                feedback = "Posture looks good!"

                # check for slouching
                # must ensure that the keypoints exist
                if 'shoulder_left' in keypoints and 'shoulder_right' in keypoints and 'hip_left' in keypoints and 'hip_right' in keypoints:
                    shoulder_left = keypoints['shoulder_left']
                    shoulder_right = keypoints['shoulder_right']
                    hip_left = keypoints['hip_left']
                    hip_right = keypoints['hip_right']

                    # Calculate angles
                    shoulder_angle = calculate_angle(shoulder_left, shoulder_right, hip_left)
                    hip_angle = calculate_angle(hip_left, hip_right, shoulder_left)

                    back_angle = calculate_angle(shoulder_left, shoulder_right, hip_left)   
                    print(f"Shoulder angle: {shoulder_angle}, Hip angle: {hip_angle}, Back angle: {back_angle}")
                    # Check for slouching
                    if shoulder_angle < 150:
                        feedback = "Slouching detected! Please straighten your back."
                    elif hip_angle < 150:
                        feedback = "Hips are misaligned! Please adjust your posture."

                # Send feedback to the client
                serialized_keypoints = {k: [int(v[0]), int(v[1])] for k, v in keypoints.items()}
                await websocket.send_text(json.dumps({"keypoints": serialized_keypoints, "feedback": feedback}))
    except Exception as e:
        print("Error processing image:", e)
        await websocket.send_text("Error processing image: " + str(e))
    finally:
        await websocket.close()
        print("WebSocket connection closed")