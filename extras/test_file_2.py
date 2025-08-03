import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if cv2.waitKey(1) and 0xFF==ord('q'):
        break

    cv2.imshow("Camera Feed", frame)

cv2.destroyAllWindows()
cap.release()
