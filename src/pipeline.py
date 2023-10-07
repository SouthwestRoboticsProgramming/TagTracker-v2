import cv2
import environment
import solve
import nt_io
import detect
import time
import gui
from dataclasses import dataclass

@dataclass
class CameraSettings:
    resolution: tuple[int, int]
    auto_exposure: int
    exposure: float
    gain: float

@dataclass
class PipelineSettings:
    camera_id: int|str
    name: str
    camera_settings: CameraSettings
    calibration: solve.CalibrationInfo
    dictionary_id: int
    enable_gui: bool = False

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

class TagTrackerPipeline:
    def __init__(self, env: environment.TagEnvironment, settings: PipelineSettings):
        self.capture = Capture(settings.camera_id, settings.camera_settings)
        self.detector = detect.TagDetector(settings.dictionary_id)
        self.estimator = solve.PoseEstimator(
            env=env,
            calibration=settings.calibration
        )
        self.io = nt_io.NetworkTablesIO(settings.name)
        self.name = settings.name
        self.enable_gui = settings.enable_gui

        self.running = True

    def run(self, gui_images: dict[str, cv2.Mat]):
        while self.running:
            frame_timestamp = time.time()
            retval, image = self.capture.read_frame()
            if not retval:
                print("did not get image")
                continue

            results = self.detector.detect(image)
            estimation = self.estimator.estimate_pose(results)
            self.io.publish_estimations(estimation, frame_timestamp)

            print(self.name + ":", estimation)

            if self.enable_gui:
                gui.overlay_image_observation(image, results)
                gui_images["Capture: " + self.name] = image
                # cv2.imshow("Capture: " + self.name, image)
                
