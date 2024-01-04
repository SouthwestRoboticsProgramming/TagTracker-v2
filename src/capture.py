import cv2
import threading
import time
import queue
from dataclasses import dataclass, field

import config

@dataclass(order=True)
class CameraFrame:
    timestamp: float
    camera: str
    calibration: config.CalibrationInfo
    image: cv2.Mat = field(compare=False)
    rate: int = field(compare=False)

class CameraInputThread(threading.Thread):
    name: str
    capture: cv2.VideoCapture
    frame_queue: queue.PriorityQueue[CameraFrame]
    fps: int
    count: int
    prev_time: float
    running: bool

    def __init__(self, settings: config.CameraSettings, frame_queue: queue.PriorityQueue[CameraFrame]):
        threading.Thread.__init__(self)
        self.name = settings.name
        # self.capture = cv2.VideoCapture(settings.id, cv2.CAP_V4L2)
        
        self.capture = cv2.VideoCapture("v4l2src device=/dev/video2 extra_controls=\"c,exposure_auto=1,exposure_absolute=10,gain=25,sharpness=0,brightness=0\" ! image/jpeg,format=MJPG,width=1600,height=1200 ! jpegdec ! video/x-raw ! appsink drop=1", cv2.CAP_GSTREAMER)
        
        self.frame_queue = frame_queue
        self.fps = 0
        self.count = 0
        self.prev_time = time.time()
        self.calibration = settings.calibration
        self.running = True

        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.calibration.resolution[0])
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.calibration.resolution[1])
        # self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, settings.auto_exposure)
        # self.capture.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        # self.capture.set(cv2.CAP_PROP_FPS, 50)
        # self.capture.set(cv2.CAP_PROP_GAIN, settings.gain)

        f = settings.stream_format
        if len(f) != 4:
            raise Exception("Stream format must be 4 chars long")
        # self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(f[0], f[1], f[2], f[3]))
    
    def run(self):
        print(self.name, "starting capture", )
        while self.running:
            timestamp = time.time()
            retval, image = self.capture.read()
            if retval:
                self.count += 1
                # Use while in case a frame took over 1 second
                while time.time() - self.prev_time > 1:
                    self.fps = self.count
                    self.count = 0
                    self.prev_time += 1
                    print(self.name, "fps:", self.fps)

                self.frame_queue.put(CameraFrame(
                    timestamp=timestamp,
                    camera=self.name,
                    calibration=self.calibration,
                    image=image,
                    rate=self.fps
                ))
            else:
                print(self.name, "did not receive image")
                time.sleep(0.2)

        self.capture.release()
        print(self.name, "stopped capture")
