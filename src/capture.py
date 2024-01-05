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

        if isinstance(settings.id, str):
            device_path = settings.id
        else:
            device_path = "/dev/video" + str(settings.id)

        pipeline = "v4l2src device=" + device_path + " extra_controls=\"c"
        pipeline += ",exposure_auto=" + str(settings.auto_exposure)
        pipeline += ",exposure_absolute=" + str(settings.exposure)
        pipeline += ",gain=" + str(settings.gain)
        pipeline += ",sharpness=0,brightness=0\""
        pipeline += " ! image/jpeg,format=MJPG"
        pipeline += ",width=" + str(settings.calibration.resolution[0])
        pipeline += ",height=" + str(settings.calibration.resolution[1])
        pipeline += " ! jpegdec ! video/x-raw ! appsink drop=1"
        print("GStreamer pipeline:", pipeline)
        
        self.capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        self.frame_queue = frame_queue
        self.fps = 0
        self.count = 0
        self.prev_time = time.time()
        self.calibration = settings.calibration
        self.running = True
    
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
