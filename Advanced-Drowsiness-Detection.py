import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Constants - adjusted for less sensitivity
EAR_THRESHOLD = 0.25  # Increase this threshold to reduce sensitivity
EAR_CONSEC_FRAMES = 120  # Require more consecutive frames for a blink
PERCLOS_THRESHOLD = 0.5  # Increase the threshold for PERCLOS
BLINK_THRESHOLD = 0.25  # Blink threshold
MAX_FRAME_HISTORY = 60  # Frames to keep in history for PERCLOS calculation
HEAD_POSE_THRESHOLD = 1.0  # Higher threshold for head tilt

# Indices for landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Initialize counters and history
COUNTER = 0
PERCLOS_HISTORY = []
BLINK_COUNT = 0
HEAD_POSE_HISTORY = []

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(shape):
    image_points = np.array([
        (shape[30, :]),     # Nose tip
        (shape[8, :]),      # Chin
        (shape[36, :]),     # Left eye left corner
        (shape[45, :]),     # Right eye right corner
        (shape[48, :]),     # Left mouth corner
        (shape[54, :])      # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),            # Nose tip
        (0.0, -330.0, -65.0),       # Chin
        (-165.0, 170.0, -135.0),    # Left eye left corner
        (165.0, 170.0, -135.0),     # Right eye right corner
        (-150.0, -150.0, -125.0),   # Left mouth corner
        (150.0, -150.0, -125.0)     # Right mouth corner
    ])

    focal_length = frame.shape[1]
    camera_matrix = np.array([
        [focal_length, 0, frame.shape[1] / 2],
        [0, focal_length, frame.shape[0] / 2],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return rotation_vector, translation_vector

def detect_blinks(ear):
    global COUNTER, BLINK_COUNT

    if ear < BLINK_THRESHOLD:
        COUNTER += 1
    else:
        if COUNTER >= EAR_CONSEC_FRAMES:
            BLINK_COUNT += 1
        COUNTER = 0

def calculate_perclos(ear_history):
    closed_frames = sum(1 for ear in ear_history if ear < EAR_THRESHOLD)
    return closed_frames / len(ear_history)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        
        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Add EAR to history for PERCLOS calculation
        if len(PERCLOS_HISTORY) >= MAX_FRAME_HISTORY:
            PERCLOS_HISTORY.pop(0)
        PERCLOS_HISTORY.append(ear)

        # Blink detection
        detect_blinks(ear)

        # Head pose estimation
        rotation_vector, translation_vector = get_head_pose(shape)
        head_tilt = np.linalg.norm(rotation_vector)

        # Smooth the head tilt using moving average
        if len(HEAD_POSE_HISTORY) >= MAX_FRAME_HISTORY:
            HEAD_POSE_HISTORY.pop(0)
        HEAD_POSE_HISTORY.append(head_tilt)
        smoothed_head_tilt = np.mean(HEAD_POSE_HISTORY)
        
        # PERCLOS calculation
        perclos = calculate_perclos(PERCLOS_HISTORY)

        # Draw landmarks
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

        # Drowsiness detection logic
        if perclos > PERCLOS_THRESHOLD and smoothed_head_tilt > HEAD_POSE_THRESHOLD:
            cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Optionally, add sound alert here
        else:
            cv2.putText(frame, f"BLINKS: {BLINK_COUNT}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
