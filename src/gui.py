import cv2
import numpy as np
from detect import DetectedTag

def overlay_image_observation(image: cv2.Mat, observations: list[DetectedTag]) -> None:
    for observation in observations:
        cv2.aruco.drawDetectedMarkers(image, np.array([observation.corners]), np.array([observation.id]))