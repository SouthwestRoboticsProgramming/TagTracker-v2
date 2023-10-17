import cv2
import numpy
import queue
import threading
import time
from dataclasses import dataclass, field

import capture
import config
import detect
import solve

@dataclass
class ProcessTimings:
    detect: float
    solve: float

@dataclass(order=True)
class FrameResult:
    frame: capture.CameraFrame
    detections: list[detect.DetectedTag] = field(compare=False)
    estimates: solve.EstimatePair = field(compare=False)
    timings: ProcessTimings = field(compare=False)

class TagProcessThread(threading.Thread):
    frame_queue: queue.PriorityQueue[capture.CameraFrame]
    result_queue: queue.PriorityQueue[FrameResult]
    detector: detect.TagDetector
    estimator: solve.PoseEstimator
    running: bool

    def __init__(self, aruco_dict: int, env: config.TagEnvironment, frame_queue: queue.PriorityQueue[capture.CameraFrame], result_queue: queue.PriorityQueue[FrameResult]):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.detector = detect.TagDetector(aruco_dict)
        self.estimator = solve.PoseEstimator(env)
        self.running = True

    def run(self):
        print("Starting process thread")
        while self.running:
            frame = None
            while frame is None:
                if not self.running:
                    break
                try:
                    frame = self.frame_queue.get(timeout=1)
                except queue.Empty as _:
                    pass
            if frame is None:
                break

            begin_time = time.time()
            detections = self.detector.detect(frame.image)
            after_detect = time.time()
            estimates = self.estimator.solve(frame.calibration, detections)
            after_solve = time.time()

            result = FrameResult(
                frame=frame,
                detections=detections,
                estimates=estimates,
                timings=ProcessTimings(
                    detect=after_detect - begin_time,
                    solve=after_solve - after_detect
                )
            )

            timings = result.timings

            # Annotate the frame with infos
            for tag in result.detections:
                cv2.aruco.drawDetectedMarkers(frame.image, numpy.array([tag.corners]), numpy.array([tag.id]))
            def put_text(text: str, pos, color):
                cv2.putText(frame.image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            put_text(frame.camera, (5, 40), (255, 255, 64))
            put_text("FPS: " + str(frame.rate), (5, 80), (64, 255, 64))
            put_text(f"Detect: {timings.detect * 1000 :.2f} ms", (5, 120), (64, 255, 64))
            put_text(f"Solve: {timings.solve * 1000 :.2f} ms", (5, 160), (64, 255, 64))

            est = result.estimates
            if est is None:
                put_text(f"No estimates this frame :(", (5, 200), (64, 255, 64))
            else:
                a = est.pose_a
                b = est.pose_b

                def format_pose(pose):
                    if pose is None:
                        return "None"
                    
                    tx = pose[0].translation()
                    return f"{tx.X():.2f}, {tx.Y():.2f}, {tx.Z():.2f} | e {pose[1]:.4f}"

                put_text(f"Est A: {format_pose(a)}", (5, 200), (64, 255, 64))
                put_text(f"Est B: {format_pose(b)}", (5, 240), (64, 255, 64))

            self.result_queue.put(result)
        print("Stopping process thread")
