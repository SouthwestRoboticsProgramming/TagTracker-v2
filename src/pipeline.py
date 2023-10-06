import cv2
from dataclasses import dataclass

@dataclass
class CameraSettings:
    opencv_id: int
    resolution: tuple[int, int]
    auto_exposure: int
    exposure: float
    gain: float

class Capture:
    video: cv2.VideoCapture

    def __init__(self, settings: CameraSettings):
        self.video = cv2.VideoCapture(settings.opencv_id)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, settings.resolution[0])
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.resolution[1])
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings.auto_exposure)
        self.video.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        self.video.set(cv2.CAP_PROP_GAIN, settings.gain)

    def read_frame(self) -> tuple[bool, cv2.Mat]:
        retval, image = self.video.read()
        return (retval, image)
