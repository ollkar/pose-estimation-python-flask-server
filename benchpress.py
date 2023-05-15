import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response, jsonify, request
import configparser

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

app = Flask(__name__)
cap = cv2.VideoCapture(0)
landmark_names = [landmark.name for landmark in mp_holistic.PoseLandmark]
current_angle: float = 0.0
length_between_wrists: float = 0.0

reps = 0
fase = None
rpe = 0
reptimer = 0.0
start_timer = 0.0


def calculate_angle(pointA, pointB, pointC):
    pointA = np.array(pointA)
    pointB = np.array(pointB)
    pointC = np.array(pointC)

    radians = np.arctan2(pointC[1]-pointB[1], pointC[0]-pointB[0]) - \
        np.arctan2(pointA[1]-pointB[1], pointA[0]-pointB[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle


def benchpress(landmarks):
    # motion of the benchpress
    # start counting when wrist point is directly above shoulder point, check i line is straight
    # when elbow point has been the midline or below and wrist and shoulder line is straight again count as 1 rep

    # Get properties
    right_shoulder = [landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_shoulder = [landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y]

    right_wrist = [landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y]
    left_wrist = [landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_WRIST.value].y]

    right_elbow = [landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_holistic.PoseLandmark.RIGHT_ELBOW.value].y]
    left_elbow = [landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_holistic.PoseLandmark.LEFT_ELBOW.value].y]

    # Get angle between right arm
    global current_angle
    current_angle = round(calculate_angle(
        right_shoulder, right_elbow, right_wrist), 0)

    # Get and calculate benchpress variation
    global benchpress_variation
    length_between_wrists = right_wrist[0] - left_wrist[0]
    benchpress_variation = "normal"

    if (length_between_wrists < -0.8):
        benchpress_variation = "wide"

    if (length_between_wrists > -0.6):
        benchpress_variation = "close"

    # Calculate reps and RPE
    # 1 rep = when elbow joint is lower than shoulder join
    global rpe
    global fase
    global reps
    global reptimer
    global start_timer

    if (current_angle > 140):
        start_timer = time.time()
        fase = "down"

    if (current_angle < 40 and fase == "down"):
        reps += 1
        reptimer = time.time() - start_timer

        # Calculate rpe
        # 1sec = rpe 1 etc..
        rpe = min((reptimer - 1) // 1 + 1, 10)

# common form mistakes
# shoulder rolled forward (up towards bar)
# wrists not stacked
# bar not touching end of chest
# bar path not correct


def generate_frames():
    # Create a holistic instance
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            else:
                # Perform any necessary image processing here

                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                # Get landmarks
                try:
                    landmarks = results.pose_landmarks.landmark
                except:
                    pass

                # Calculate angles, reps
                benchpress(landmarks)

                # Convert the frame to JPEG format and then BYTE
                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

                # Yield the frame as a response to the client
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_current_angle')
def get_current_angle():
    return Response(str(current_angle))


@app.route('/get_benchpress_variation')
def get_benchpress_variation():
    return Response(str(benchpress_variation))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
