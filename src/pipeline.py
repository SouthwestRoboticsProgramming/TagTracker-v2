import cv2
import environment
import solve
import nt_io
import detect
import time
import gui
import output_logger
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
    logger: output_logger.OutputLogger = None

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
        self.logger = settings.logger

        self.running = True

    def run(self, gui_images: dict[str, cv2.Mat]):
        fps = 0
        fps_count = 0
        fps_start = time.time()
        while self.running:
            frame_timestamp = time.time()
            retval, image = self.capture.read_frame()
            if not retval:
                # print("did not get image")
                continue

            results = self.detector.detect(image)
            estimation = self.estimator.estimate_pose(results)

            fps_count += 1
            if frame_timestamp - fps_start > 1:
                fps_start += 1
                fps = fps_count
                fps_count = 0

            self.io.publish_data(estimation, frame_timestamp, fps)

            # if estimation:
            #     print(self.name + ":", estimation)

            if self.enable_gui:
                gui.overlay_image_observation(image, results)
                gui.overlay_frame_rate(image, fps)
                gui_images["Capture: " + self.name] = image
            if self.logger:
                for detection in results:
                    self.logger.log_tag_detect(
                        frame_timestamp=frame_timestamp,
                        cam=self.name,
                        tag_id=detection.id,
                        corners=detection.corners
                    )
                if estimation:
                    self.logger.log_estimate(frame_timestamp, self.name, estimation)
                
