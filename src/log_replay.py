import queue
import detect
import time
import output_logger
import struct
import numpy
from dataclasses import dataclass

class LogTagDetector:
    def __init__(self, replay):
        self.replay = replay
        self.queue = queue.Queue()

    def submit(self, data: list[detect.DetectedTag]):
        self.queue.put(data)

    def detect(self) -> tuple[list[detect.DetectedTag], None]: 
        data = self.queue.get(block=True, timeout=None) # Wait until get data
        return [data, None]

@dataclass
class LogEvent:
    timestamp: float
    id: int
    cam: str
    data: bytes

class ByteStream:
    def __init__(self, data: bytes):
        self.data = data
        self.idx = 0
    
    def next(self, count: int) -> bytes:
        vals = self.data[self.idx : (self.idx + count)]
        self.idx += count
        return vals

class LogReplay:
    def __init__(self, file_name: str, real_time=True):
        self.file = open(file_name, "rb")
        print("Replaying", file_name, "| Real time:", real_time)
        input("Press Enter to begin replay...")

        self.queued_event = self.read_event()
        self.start_timestamp = self.queued_event.timestamp
        self.start_time = time.time()

        self.detectors = {}

    def get_detector(self, cam_name: str):
        detector = LogTagDetector(self)
        self.detectors[cam_name] = detector
        return detector

    def read_event(self):
        b = self.file.read(1)
        if b == b'':
                print("Replay finished")
                return None # Hit EOF - log ended
        
        while b[0] != output_logger.START_BYTE:
            b = self.file.read(1)
            if b == b'':
                print("Replay finished")
                return None # Hit EOF - log ended
        
        len = struct.unpack(">H", self.file.read(2))[0]
        data = self.file.read(len)

        frame_timestamp, event_id, cam_len = struct.unpack(">dbb", data[0:10])
        cam = data[10 : (cam_len + 10)].decode()
        event_data = data[cam_len + 10:]

        return LogEvent(
            timestamp=frame_timestamp,
            id=event_id,
            cam=cam,
            data=event_data
        )
    
    def dispatch_tag_detects(self, event: LogEvent):
        d = ByteStream(event.data)

        detect_count = struct.unpack(">b", d.next(1))[0]
        detections = []
        for i in range(detect_count):
            id, corner_count = struct.unpack(">bb", d.next(2))
            corners = []
            for j in range(corner_count):
                x, y = struct.unpack(">dd", d.next(16))
                corners.append([x, y])
            detections.append(detect.DetectedTag(
                id,
                numpy.array([corners])
            ))

        print("Replay: detect", detections)

        if event.cam in self.detectors:
            detector = self.detectors[event.cam]
            detector.submit(detections)

    def step(self):
        elapsed_time = time.time() - self.start_time
        replay_pos = self.start_timestamp + elapsed_time

        while self.queued_event is not None and self.queued_event.timestamp <= replay_pos:
            # Dispatch self.queued_event
            if self.queued_event.id == 0:
                self.dispatch_tag_detects(self.queued_event)

            self.queued_event = self.read_event()

        # now = time.time()
        # while not self.has_started:
        #     # Find the start byte
        #     b = self.file.read(1)
        #     while b != output_logger.START_BYTE:
        #         b = self.file.read(1)

        #     len = struct.unpack(">H", self.file.read(2))
        #     data = self.file.read(len)

        #     frame_timestamp, event_id, cam_len = struct.unpack(">dbb", data[0:10])

        #     if not self.has_started:
        #         self.start_timestamp = frame_timestamp
        #         self.start_time = time.time()

        #     cam = data[10:(cam_len+10)].decode()

