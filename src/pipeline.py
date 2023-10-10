import cv2
import environment
import solve
import nt_io
import detect
import time
import gui
import output_logger
import log_replay
from dataclasses import dataclass

@dataclass
class PipelineSettings:
    camera_id: int|str
    name: str
    camera_settings: detect.CameraSettings
    calibration: solve.CalibrationInfo
    dictionary_id: int
    enable_gui: bool = False
    logger: output_logger.OutputLogger = None
    replay: log_replay.LogReplay = None

class TagTrackerPipeline:
    def __init__(self, env: environment.TagEnvironment, settings: PipelineSettings):
        if settings.replay:
            self.detector = settings.replay.get_detector(settings.name)
        else:
            self.detector = detect.TagDetector(settings.dictionary_id, settings.camera_id, settings.camera_settings)
        
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
        print("Capture started:", self.name)
        while self.running:
            frame_timestamp = time.time()

            results, image = self.detector.detect()
            estimation = self.estimator.estimate_pose(results)

            fps_count += 1
            if frame_timestamp - fps_start > 1:
                fps_start += 1
                fps = fps_count
                fps_count = 0

            self.io.publish_data(estimation, frame_timestamp, fps)

            # if estimation:
            #     print(f"{self.name} : {estimation}")

            if self.enable_gui and image is not None:
                gui.overlay_image_observation(image, results)
                gui.overlay_frame_rate(image, fps)
                gui_images["Capture: " + self.name] = image
            if self.logger:
                if len(results) > 0:
                    self.logger.log_tag_detects(frame_timestamp, self.name, results)
                if estimation:
                    self.logger.log_estimate(frame_timestamp, self.name, estimation)
                
