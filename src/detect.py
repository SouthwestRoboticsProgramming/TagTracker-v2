import cv2
import numpy
import numpy.typing
from dataclasses import dataclass

@dataclass
class CameraSettings:
    resolution: tuple[int, int]
    auto_exposure: int
    exposure: float
    gain: float

@dataclass
class DetectedTag:
    id: int
    corners: numpy.typing.NDArray[numpy.float64]

class Capture:
    video: cv2.VideoCapture

    def __init__(self, camera_id: int|str, settings: CameraSettings):
        self.video = cv2.VideoCapture(camera_id)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings.auto_exposure)
        self.video.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        self.video.set(cv2.CAP_PROP_GAIN, settings.gain)

    def read_frame(self) -> tuple[bool, cv2.Mat]:
        retval, image = self.video.read()
        return (retval, image)

class TagDetector:
    def __init__(self, dictionary, camera_id: int|str, settings: CameraSettings):
        dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self.capture = Capture(camera_id, settings)
        self.detector = cv2.aruco.ArucoDetector(dict, params)

    def detect(self) -> tuple[list[DetectedTag], cv2.Mat]:
        retval, image = self.capture.read_frame()
        if not retval:
            print("Did not receive image!")
            return ([], None)

        corners, ids, _ = self.detector.detectMarkers(image)
        if len(corners) == 0:
            return ([], image)
        return ([DetectedTag(id[0], corner) for id, corner in zip(ids, corners)], image)

