import streamlit as st
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from streamlit_extras.camera_input_live import camera_input_live

SWIPE_ANGLE_THRESHOLD = 30
SWIPE_TIME_WINDOW = 0.32
RECALL_MEMORY = 16

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawings_styles = mp.solutions.drawing_styles

angle_history = deque(maxlen=RECALL_MEMORY)

hand_detector = mp_hands.Hands(
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    max_num_hands=1
)

def get_center(hand_landmarks, scale_vector=None, return_type=float):
    cx = (hand_landmarks[9].x + hand_landmarks[0].x) / 2
    cy = (hand_landmarks[9].y + hand_landmarks[0].y) / 2
    if scale_vector is not None:
        cx *= scale_vector[0]
        cy *= scale_vector[1]
    return return_type(cx), return_type(cy)

def draw_landmarks(frame, hand):
    mp_drawings.draw_landmarks(
        frame,
        hand,
        mp_hands.HAND_CONNECTIONS,
        mp_drawings.DrawingSpec(color=(25, 50, 12), thickness=2),
        mp_drawings.DrawingSpec(color=(255, 0, 36), thickness=2)
    )
    cv2.circle(frame, get_center(hand.landmark, (fw, fh), return_type=int), 5, (126, 81, 63), -1)
    cv2.circle(frame, get_center(hand.landmark, (fw, fh), return_type=int), 6, (25, 45, 36), 1)

def norm(vec):
    x, y = vec
    return np.sqrt(x ** 2 + y ** 2)

def calculate_angle_and_direction_to_perp(hand_landmark):
    cx, cy = get_center(hand_landmark)
    mx, my = hand_landmark[9].x, hand_landmark[9].y

    x = mx - cx
    y = my - cy
    mod_xy = norm((x, y))
    direction = 1 if x > 0 else -1

    cos_theta = -y / mod_xy if mod_xy != 0 else 0
    angle = np.degrees(np.arccos(cos_theta))

    return direction, angle

def detect_swipe(angle_history):
    if len(angle_history) < 2:
        return False, None
    latest_timestamp, latest_direction, latest_angle = angle_history[-1]
    for i in range(len(angle_history) - 1):
        older_timestamp, older_direction, older_angle = angle_history[i]
        if latest_timestamp - older_timestamp >= SWIPE_TIME_WINDOW:
            angle_diff = abs(older_angle - latest_angle)
            direction_change = latest_direction != older_direction
            if angle_diff >= SWIPE_ANGLE_THRESHOLD and direction_change:
                return True, latest_direction
    return False, None

def reset_history(angle_history):
    latest_history = angle_history[-1]
    angle_history = deque(maxlen=RECALL_MEMORY)
    angle_history.append(latest_history)
    return angle_history

st.title("Hand Gesture Swipe Detector")
frame = camera_input_live()

if frame:
    # Convert PIL Image to OpenCV BGR format
    image_np = np.array(frame)
    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    fh, fw, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_detector.process(rgb_image)
    landmarks = results.multi_hand_landmarks

    if landmarks:
        for hand in landmarks:
            draw_landmarks(frame, hand)
            direction, angle = calculate_angle_and_direction_to_perp(hand.landmark)
            time_stamp = time.time()

            angle_history.append((time_stamp, direction, angle))
            swipe_detected, direction_of_swipe = detect_swipe(angle_history)

            if swipe_detected:
                angle_history = reset_history(angle_history)
                st.success(f"Swipe: {'Right' if direction_of_swipe == 1 else 'Left'}")

    # Display the frame in Streamlit
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live Feed")
