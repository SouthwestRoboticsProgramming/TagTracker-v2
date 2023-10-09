import numpy.typing
import os
from wpimath.geometry import *
import struct
import queue
import threading
import time

import solve

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

class OutputLogger:
    def __init__(self, file_name: str, also_print: bool = False):
        self.write_queue = queue.Queue()
        thr = threading.Thread(target=write_thread, args=(file_name, self.write_queue))
        thr.daemon = True
        thr.start()
        self.should_print = also_print

    def log_tag_detect(
            self,
            frame_timestamp: float,
            cam: str, 
            tag_id: int, 
            corners: numpy.typing.NDArray[numpy.float64]):
        data = pack_header(frame_timestamp, 0, cam)

        # Pretty sure it's safe to assume there's always 4 corners, but
        # prefixing with count anyway
        data += struct.pack(">bb", tag_id, len(corners[0]))
        for corner in corners[0]:
            data += struct.pack(">dd", corner[0], corner[1])

        self.write_queue.put(data)
        if self.should_print:
            corners_str = ""
            for corner in corners[0]:
                corners_str += " " + str(corner[0]) + "," + str(corner[1])
            print(frame_timestamp, cam, "detect:", tag_id, "@" + corners_str)

    def log_estimate(
            self,
            frame_timestamp: float,
            cam: str,
            estimate: solve.PoseEstimation):
        data = pack_header(frame_timestamp, 1, cam) + struct.pack(">b", len(estimate.ids))
        for id in estimate.ids:
            data += struct.pack(">b", id)

        data += pack_estimate(estimate.estimate_a)
        if estimate.estimate_b:
            data += struct.pack('>?', True)
            data += pack_estimate(estimate.estimate_b)
        else:
            data += struct.pack('>?', False)

        # self.file.write(data)
        self.write_queue.put(data)

        if self.should_print:
            print(frame_timestamp, cam, "est:", estimate)
