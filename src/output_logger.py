import os
import queue
import struct
import threading
import time
from wpimath.geometry import *

import detect
import nt_io

START_BYTE = 0x5A

def pack_header(timestamp: float, event_id: int, cam: str) -> bytes:
    cam_data = cam.encode()
    data = struct.pack(">bdbb", START_BYTE, timestamp, event_id, len(cam_data))
    data += cam_data
    return data

def pack_estimate(est: tuple[Pose3d, float]) -> bytes:
    pose, err = est

    t = pose.translation()
    q = pose.rotation().getQuaternion()
    tx = t.X()
    ty = t.Y()
    tz = t.Z()
    qw = q.W()
    qx = q.X()
    qy = q.Y()
    qz = q.Z()

    return struct.pack(">dddddddd", err, tx, ty, tz, qw, qx, qy, qz)

def write_thread(file_name: str, write_queue: queue.Queue):
    file = open(file_name, "wb")

    last_flush = time.time()
    while True:
        now = time.time()
        if now - last_flush >= 1:
            # Write data to filesystem cache
            file.flush()
            # Write filesystem cache to physical disk
            os.fsync(file.fileno())

        try:
            to_write = write_queue.get(block=True, timeout=1)
            file.write(to_write)
        except queue.Empty:
            # Ignore that
            pass

class FileLogger:
    def __init__(self, file_name: str):
        print("Logging to " + file_name)
        self.write_queue = queue.Queue()
        thr = threading.Thread(target=write_thread, args=(file_name, self.write_queue))
        thr.daemon = True
        thr.start()

    def write_event(self, frame_timestamp: float, event_id: int, cam: str, data: bytes):
        cam_data = cam.encode()
        header = struct.pack(">dbb", frame_timestamp, event_id, len(cam_data))
        header += cam_data
        header += data
        
        d = struct.pack(">bH", START_BYTE, len(header))
        d += header
        self.write_queue.put(d)

    def log_tag_detects(
            self,
            frame_timestamp: float,
            cam: str, 
            detections: list[detect.DetectedTag]):
        data = struct.pack(">b", len(detections))
        for detect in detections:
            tag_id = detect.id
            corners = detect.corners

            # Pretty sure it's safe to assume there's always 4 corners, but
            # prefixing with count anyway
            data += struct.pack(">bb", tag_id, len(corners[0]))
            for corner in corners[0]:
                data += struct.pack(">dd", corner[0], corner[1])

        self.write_event(frame_timestamp, 0, cam, data)

    def log_match_info(self, info: nt_io.MatchInfo):
        event_data = info.event_name.encode()
        data = struct.pack(">H", len(event_data)) + event_data

        data += struct.pack(
            ">iii?i",
            info.match_num,
            info.match_type,
            info.replay_num,
            info.is_red,
            info.station_num
        )

        self.write_event(time.time(), 1, "", data)
