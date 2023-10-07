import cv2
import numpy as np
from detect import DetectedTag

def overlay_image_observation(image: cv2.Mat, observations: list[DetectedTag]) -> None:
    for observation in observations:
        cv2.aruco.drawDetectedMarkers(image, np.array([observation.corners]), np.array([observation.id]))

def overlay_frame_rate(image: cv2.Mat, frame_rate: float):
    cv2.putText(image, "FPS: " + str(frame_rate), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 255, 64), 2)