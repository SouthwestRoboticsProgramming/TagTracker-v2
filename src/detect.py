import cv2
import numpy
import numpy.typing
from dataclasses import dataclass

@dataclass
class DetectedTag:
    id: int
    corners: numpy.typing.NDArray[numpy.float64]

class TagDetector:
    def __init__(self, dictionary):
        dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dict, params)

    def detect(self, image: cv2.Mat) -> list[DetectedTag]:
        corners, ids, _ = self.detector.detectMarkers(image)
        if len(corners) == 0:
            return []
        return [DetectedTag(id[0], corner) for id, corner in zip(ids, corners)]

