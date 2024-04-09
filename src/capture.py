import cv2
import threading
import time
import queue
from dataclasses import dataclass, field

import config

@dataclass
class CameraParams:
    auto_exposure: bool
    exposure: int
    gain: int
    target_fps: int

@dataclass(order=True)
class CameraFrame:
    timestamp: float
    camera: str
    calibration: config.CalibrationInfo
    image: cv2.Mat = field(compare=False)
    rate: int = field(compare=False)

def is_config_different(a: CameraParams, b: CameraParams) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    
    return a.auto_exposure != b.auto_exposure or a.exposure != b.exposure or a.gain != b.gain or a.target_fps != b.target_fps

class CameraInputThread(threading.Thread):
    name: str
    capture: cv2.VideoCapture
    frame_queue: queue.PriorityQueue[CameraFrame]
    fps: int
    count: int
    prev_time: float
    running: bool

    # nt is CameraNetworkTablesIO
    def __init__(self, settings: config.CameraSettings, frame_queue: queue.PriorityQueue[CameraFrame], nt):
        threading.Thread.__init__(self)
        self.settings = settings
        self.nt = nt
        
        self.frame_queue = frame_queue
        self.fps = 0
        self.count = 0
        self.prev_time = time.time()
        self.calibration = settings.calibration
        self.capture = None
        self.current_config = None
        self.running = True

    def next_frame(self) -> tuple[bool, cv2.Mat]:
        config = self.nt.get_config_params()
        if is_config_different(self.current_config, config) and self.capture != None:
            print(self.settings.name, "stopping capture")
            self.capture.release()
            self.capture = None
        
        if self.capture is None and config != None:
            print(self.settings.name, "opening capture")
            self.capture = cv2.VideoCapture(self.settings.id, cv2.CAP_V4L2)
        
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.calibration.resolution[0])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calibration.resolution[1])
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            self.capture.set(cv2.CAP_PROP_FPS, config.target_fps)
            self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if config.auto_exposure else 1)
            self.capture.set(cv2.CAP_PROP_EXPOSURE, config.exposure)
            self.capture.set(cv2.CAP_PROP_GAIN, config.gain)

            self.current_config = config
            print(self.settings.name, "applied config:", config)

        if self.capture is None:
            return (False, None)
        else:
            ret = self.capture.read()
            if not ret[0]:
                print(self.settings.name, "did not receive image")
            return ret

    def run(self):
        print(self.settings.name, "starting capture thread")
        while self.running:
            timestamp = time.time()
            retval, image = self.next_frame()
            if retval:
                self.count += 1
                # Use while in case a frame took over 1 second
                while time.time() - self.prev_time > 1:
                    self.fps = self.count
                    self.count = 0
                    self.prev_time += 1
                    print(self.settings.name, "fps:", self.fps)

                self.frame_queue.put(CameraFrame(
                    timestamp=timestamp,
                    camera=self.settings.name,
                    calibration=self.calibration,
                    image=image,
                    rate=self.fps
                ))
            else:
                time.sleep(1)

        self.capture.release()
        print(self.settings.name, "stopped capture thread")
