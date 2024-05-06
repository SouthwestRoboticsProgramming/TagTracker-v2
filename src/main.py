# Data flow:
# Camera input threads put frames into frame queue
# Tag process threads take frames from frame queue, find tags, estimate pose, put results into result queue
# Main thread reads result queue, sends to NT, logs, shows GUI

import cv2
import math
import queue
import random
import time
from argparse import ArgumentParser

import config
import capture
import nt_io
import output_logger
import process
import web_stream

def main():
    parser = ArgumentParser(
        prog="TagTracker-v2",
        description="AprilTag tracker for FRC"
    )
    parser.add_argument("-g", "--gui", action="store_true", help="Enable camera preview GUI")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to config JSON")
    # parser.add_argument("-r", "--replay", type=str, help="Log file to replay")
    # parser.add_argument("-f", "--fast", action="store_true", help="Replay faster than real time")
    args = parser.parse_args()

    conf = config.load_config(args.config)

    if conf.tag_family == "16h5":
        dict = cv2.aruco.DICT_APRILTAG_16H5
    elif conf.tag_family == "36h11":
        dict = cv2.aruco.DICT_APRILTAG_36H11
    else:
        print("Unsupported tag family: " + conf.tag_family)
        return

    # Connect to NetworkTables server
    nt = nt_io.NetworkTablesIO(conf.networktables, conf.environment)

    frame_queue = queue.PriorityQueue()
    result_queue = queue.PriorityQueue()

    threads = []
    for camera_config in conf.cameras:
        io = nt.get_camera_io(camera_config.name)
        res = camera_config.calibration.resolution
        io.publish_image_resolution(int(res[0]), int(res[1]))
        threads.append(capture.CameraInputThread(camera_config, frame_queue, io))

    for _ in range(0, conf.process_threads):
        threads.append(process.TagProcessThread(dict, conf.environment, frame_queue, result_queue))

    stream = web_stream.StreamServer(conf.stream)
    stream.start()

    if conf.logging.enabled:
        log_file_name = conf.logging.output_dir + "log_" + str(math.floor(random.random() * 1e16)) + ".ttlog"
        logger = output_logger.FileLogger(log_file_name)
    else:
        logger = None

    for thread in threads:
        thread.start()

    try:
        camera_count = len(conf.cameras)
        i = 0
        prev_match_info = None
        while True:
            result = result_queue.get()
            frame = result.frame

            top = frame.image.shape[0] - 100
            def put_text(text: str, pos, color):
                cv2.putText(frame.image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            put_text("Frame queue: " + str(frame_queue.qsize()), (5, top), (64, 128, 255))
            put_text("Result queue: " + str(result_queue.qsize()), (5, top + 40), (64, 128, 255))
            put_text(f"Frame age: {(time.monotonic() - frame.timestamp) :.3f}", (5, top + 80), (64, 128, 255))

            nt.publish_output(result)
            stream.publish_frame(frame.camera, frame.image)

            if logger:
                match_info = nt.get_match_info()
                if match_info != prev_match_info:
                    print("Got match info:", match_info)
                    logger.log_match_info(match_info)
                    prev_match_info = match_info
                
                if len(result.detections) != 0:
                    logger.log_tag_detects(frame.timestamp, frame.camera, result.detections)

            if args.gui:
                cv2.imshow(frame.camera, frame.image)
                if i == 0 and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Only waitKey once for all cameras
            i += 1
            if i >= camera_count:
                i = 0
    except KeyboardInterrupt as _:
        print("Interrupted...")

    for thread in threads:
        thread.running = False
    for thread in threads:
        thread.join()

    print("done :)")

if __name__ == "__main__":
    main()
