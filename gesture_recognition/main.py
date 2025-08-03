import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque

import pyautogui

SWIPE_ANGLE_THRESHOLD = 30 # Angle in degrees
SWIPE_TIME_WINDOW = 0.3 # Time in seconds
RECALL_MEMORY = 20
MINIMUM_TIME_GAP_BETWEEN_SWIPES = 1.7 # Time in seconds

mp_hands = mp.solutions.hands
mp_drawings = mp.solutions.drawing_utils
mp_drawings_styles = mp.solutions.drawing_styles

angle_history = deque(maxlen=RECALL_MEMORY)
last_swiped = 0.0

cap = cv2.VideoCapture(0)
hand_detector = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)

def get_center(hand_landmarks, scale_vector=None, return_type=float):
    cx = (hand_landmarks[9].x + hand_landmarks[0].x)/2
    cy = (hand_landmarks[9].y + hand_landmarks[0].y)/2
    if scale_vector is not None:
        cx *= scale_vector[0]
        cy *= scale_vector[1]

    return return_type(cx), return_type(cy)

def draw_landmarks(frame, hand):
    mp_drawings.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, mp_drawings_styles.DrawingSpec((25, 50, 12)), mp_drawings_styles.DrawingSpec((255, 0, 36)))

    # Main Circle around the center of my hand
    cv2.circle(frame, get_center(hand.landmark, (fw, fh), return_type=int), 5, (126, 81, 63), -1)
    # Boundary around the center point of palm
    cv2.circle(frame, get_center(hand.landmark, (fw, fh), return_type=int), 6, (25, 45, 36), 1)

def norm(vec):
    x, y = vec
    return np.sqrt(x**2+y**2)

def calculate_angle_and_direction_to_perp(hand_landmark):
    cx, cy = get_center(hand_landmark)
    mx, my = hand_landmark[9].x, hand_landmark[9].y

    # norm_mid = norm_the_vec(np.array([mx-cx, my-cy]))
    # perp_vec = np.array([0, -1])
    # return np.degrees(np.arccos(np.dot(norm_mid, perp_vec)))

    x = (mx-cx)
    y = (my-cy)

    mod_xy = norm((x, y))
    direction = np.where(x>0, 1, -1)

    cos_theta = -y/mod_xy if mod_xy!=0 else 0
    angle = np.degrees(np.arccos(cos_theta))

    return direction, angle


def detect_swipe(angle_history):
    if len(angle_history)<2:
        return False, None

    latest_timestamp, latest_direction, latest_angle = angle_history[-1]
    for i in range(len(angle_history)-1):
        older_timestamp, older_direction, older_angle = angle_history[i]

        if latest_timestamp-older_timestamp>=SWIPE_TIME_WINDOW:
            angle_diff = abs(older_angle-latest_angle)
            direction_change = latest_direction != older_direction
            if angle_diff >= SWIPE_ANGLE_THRESHOLD and direction_change:
                return True, latest_direction

    return False, None

def reset_history(angle_history):
    # latest_history = angle_history[-1]
    angle_history = deque(maxlen=RECALL_MEMORY)
    # angle_history.append(latest_history)
    return angle_history

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1) # (dfs)

    fh, fw, _ = frame.shape
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hand_detector.process(rgb_image)
    landmarks = results.multi_hand_landmarks

    if landmarks:
        no_of_hands_detected = len(landmarks)
        for hand_idx, hand in enumerate(landmarks):
            draw_landmarks(frame, hand)
            direction, angle = calculate_angle_and_direction_to_perp(hand.landmark)
            time_stamp = time.time()

            angle_history.append((time_stamp, direction, angle))
            swipe_detected, direction_of_swap = detect_swipe(angle_history)

            if swipe_detected and  time.time()-last_swiped>=MINIMUM_TIME_GAP_BETWEEN_SWIPES:
                angle_history = reset_history(angle_history)
                print(f"Swipe:{'Right' if direction_of_swap==1 else 'Left'}")
                if direction_of_swap>0:
                    pyautogui.click(x=800, y=550)
                elif direction_of_swap<0:
                    pyautogui.click(x=700, y=550)
                last_swiped = time.time()


    key = cv2.waitKey(1)
    if key == ord("q"):
        break

    cv2.imshow("Image", frame)