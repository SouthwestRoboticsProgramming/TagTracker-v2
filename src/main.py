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

def main():
    # Load tag environment from WPILib json
    env = environment.TagEnvironment("environment.json")

    # Connect to NetworkTables server
    nt = ntcore.NetworkTableInstance.getDefault()
    nt.setServer("localhost")
    nt.startClient4("TagTracker-v2")
    
    # Publish the tag environment so ShuffleLog can visualize it
    env_pub = nt.getDoubleArrayTopic("/TagTracker/environment").publish()
    env_data = []
    for id, pose in env.tags.items():
        env_data.append(id)
        nt_io.append_pose(env_data, pose)
    env_pub.set(env_data)

    logger = output_logger.OutputLogger("log-" + str(math.floor(time.time() * 1000000)) + ".log", also_print=True)

    # Set up the pipeline
    webcam_pipe = pipeline.TagTrackerPipeline(
        env=env,
        settings=pipeline.PipelineSettings(
            camera_id=0,
            name="laptop",
            camera_settings=pipeline.CameraSettings(
                resolution=[640, 480],
                auto_exposure=3,
                exposure=0.01,
                gain=1
            ),
            calibration=solve.CalibrationInfo(
                matrix=numpy.array([
                    676.6192195641298, 0, 385.1137834870396,
                    0, 676.8359339562655, 201.81402152233636,
                    0, 0, 1
                ]).reshape(3, 3),
                distortion_coeffs=numpy.array([0.01632932, -0.36390723, -0.01638719,  0.02577886,  0.93133364])
            ),
            dictionary_id=cv2.aruco.DICT_APRILTAG_16H5,
            enable_gui=True,
            logger=logger
        )
    )
    usb_pipe = pipeline.TagTrackerPipeline(
        env=env,
        settings=pipeline.PipelineSettings(
            camera_id=2,
            name="usb",
            camera_settings=pipeline.CameraSettings(
                resolution=[640, 480],
                auto_exposure=3,
                exposure=0.01,
                gain=1
            ),
            calibration=solve.CalibrationInfo(
                matrix=numpy.array([
                    676.6192195641298, 0, 385.1137834870396,
                    0, 676.8359339562655, 201.81402152233636,
                    0, 0, 1
                ]).reshape(3, 3),
                distortion_coeffs=numpy.array([0.01632932, -0.36390723, -0.01638719,  0.02577886,  0.93133364])
            ),
            dictionary_id=cv2.aruco.DICT_APRILTAG_16H5,
            enable_gui=True,
            logger=logger
        )
    )

    pipelines = [webcam_pipe, usb_pipe]

    # Run the pipelines
    gui_images = {}
    threads = [threading.Thread(target=pipe.run, args=(gui_images,)) for pipe in pipelines]
    for thread in threads:
        thread.daemon = True
        thread.start()
    
    # Wait for threads to finish and show gui images when they are available
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

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Stop the pipeline threads
    for pipe in pipelines:
        pipe.running = False
    for thread in threads:
        thread.join()
    
    cv2.destroyAllWindows()

    # ------------

    # settings = pipeline.CameraSettings(
    #     opencv_id=2,
    #     resolution=[640, 480],
    #     auto_exposure=3,
    #     exposure=0.01,
    #     gain=1
    # )
    # capture = pipeline.Capture(settings)
    # detector = detect.TagDetector(cv2.aruco.DICT_APRILTAG_16H5)
    # estimator = solve.PoseEstimator(
    #     tag_size=0.15301, # Tag outer size in meters
    #     env=env,

    #     # Make up some arbitrary calibration data (very wrong)
    #     calibration=solve.CalibrationInfo(
    #         matrix=numpy.array([
    #             676.6192195641298, 0, 385.1137834870396,
    #             0, 676.8359339562655, 201.81402152233636,
    #             0, 0, 1
    #         ]).reshape(3, 3),
    #         distortion_coeffs=numpy.array([0.01632932, -0.36390723, -0.01638719,  0.02577886,  0.93133364])
    #     )
    # )
    # io = nt_io.NetworkTablesIO("localhost", "TagTracker-v2", env)

    # while True:
    #     # TODO: Is it better to sample timestamp before or after reading frame?
    #     frame_timestamp = time.time()
    #     retval, image = capture.read_frame()

    #     if retval:
    #         results = detector.detect(image)
    #         estimation = estimator.estimate_pose(results)
    #         # print(estimation)

    #         gui.overlay_image_observation(image, results)
    #         io.publish_estimations(estimation, frame_timestamp)

    #         cv2.imshow("capture", image)
    #     else:
    #         print("did not get image")
        
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
