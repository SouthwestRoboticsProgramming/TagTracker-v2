import pipeline
import solve
import cv2
import numpy
import environment
import nt_io
import ntcore
import threading
import output_logger
import time
import math
import sys
import json
from dataclasses import dataclass
from argparse import ArgumentParser

@dataclass
class CalibrationData:
    resolution: tuple[float, float]
    info: solve.CalibrationInfo

def load_calibration_data(file_name: str):
    with open(file_name, 'r') as json_file:
        json_obj = json.load(json_file)

    res_obj = json_obj["camera_resolution"]
    res_data = res_obj["data"]
    res = (res_data[1], res_data[0])

    mtx_obj = json_obj["camera_matrix"]
    mtx = numpy.array(mtx_obj["data"]).reshape(3, 3)

    dist_obj = json_obj["distortion_coefficients"]
    dist = numpy.array(dist_obj["data"])

    return CalibrationData(
        resolution=res,
        info=solve.CalibrationInfo(
            matrix=mtx,
            distortion_coeffs=dist
        )
    )

def main():
    parser = ArgumentParser(
        prog="TagTracker-v2",
        description="AprilTag tracker for FRC"
    )
    parser.add_argument("-g", "--gui", action="store_true", help="Enable camera preview GUI")
    parser.add_argument("-c", "--config", type=str, default="config.json", help="Path to config JSON")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_obj = json.load(f)

    # Load tag environment from WPILib json
    env = environment.TagEnvironment(config_obj["environment"])

    if config_obj["tag-family"] == "16h5":
        dict = cv2.aruco.DICT_APRILTAG_16H5
    elif config_obj["tag-family"] == "36h11":
        dict = cv2.aruco.DICT_APRILTAG_36H11
    else:
        print("Unsupported tag family: " + str(config_obj["tag-family"]))
        return

    # Connect to NetworkTables server
    nt_config = config_obj["networktables"]
    nt = ntcore.NetworkTableInstance.getDefault()
    nt.setServer(nt_config["server-ip"])
    nt.startClient4(nt_config["identity"])
    
    # Publish the tag environment so ShuffleLog can visualize it
    env_pub = nt.getDoubleArrayTopic("/TagTracker/environment").publish()
    env_data = []
    for id, pose in env.tags.items():
        env_data.append(id)
        nt_io.append_pose(env_data, pose)
    env_pub.set(env_data)
    
    if config_obj["log-outputs"]:
        logger = output_logger.OutputLogger("log-" + str(math.floor(time.time() * 1000000)) + ".ttlog", also_print=True)
    else:
        logger = None

    # Set up the pipelines
    pipelines = []
    for camera_obj in config_obj["cameras"]:
        calib = load_calibration_data(camera_obj["calibration"])
        pipe = pipeline.TagTrackerPipeline(
            env=env,
            settings=pipeline.PipelineSettings(
                camera_id=camera_obj["id"],
                name=camera_obj["name"],
                camera_settings=pipeline.CameraSettings(
                    resolution=calib.resolution,
                    auto_exposure=camera_obj["auto-exposure"],
                    exposure=camera_obj["exposure"],
                    gain=camera_obj["gain"]
                ),
                calibration=calib.info,
                dictionary_id=dict,
                enable_gui=args.gui,
                logger=logger
            )
        )
        pipelines.append(pipe)

    # Run the pipelines
    gui_images = {}
    threads = [threading.Thread(target=pipe.run, args=(gui_images,)) for pipe in pipelines]
    for thread in threads:
        thread.daemon = True
        thread.start()

    # Wait for threads to finish and show gui images when they are available
    try:
        while True:
            # Show images on main thread
            # OpenCV crashes if you do this on the pipeline thread
            for name, image in gui_images.copy().items():
                cv2.imshow(name, image)

            # Stop everything if a thread dies (for debugging)
            # FIXME: Disable this in competition code!
            for thread in threads:
                if not thread.is_alive():
                    print("A thread crashed, stopping...")
                    sys.exit(1)

            if args.gui and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Shutting down...")
        
    # Stop the pipeline threads
    for pipe in pipelines:
        pipe.running = False
    for thread in threads:
        thread.join()
    
    if args.gui:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
