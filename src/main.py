import pipeline
import detect
import solve
import cv2
import numpy
import gui
import environment
import time
import nt_io

def main():
    env = environment.TagEnvironment("environment.json")
    settings = pipeline.CameraSettings(
        opencv_id=2,
        resolution=[640, 480],
        auto_exposure=3,
        exposure=0.01,
        gain=1
    )
    capture = pipeline.Capture(settings)
    detector = detect.TagDetector(cv2.aruco.DICT_APRILTAG_16H5)
    estimator = solve.PoseEstimator(
        tag_size=0.15301, # Tag outer size in meters
        env=env,

        # Make up some arbitrary calibration data (very wrong)
        calibration=solve.CalibrationInfo(
            matrix=numpy.array([
                676.6192195641298, 0, 385.1137834870396,
                0, 676.8359339562655, 201.81402152233636,
                0, 0, 1
            ]).reshape(3, 3),
            distortion_coeffs=numpy.array([0.01632932, -0.36390723, -0.01638719,  0.02577886,  0.93133364])
        )
    )
    io = nt_io.NetworkTablesIO("localhost", "TagTracker-v2", env)

    while True:
        # TODO: Is it better to sample timestamp before or after reading frame?
        frame_timestamp = time.time()
        retval, image = capture.read_frame()

        if retval:
            results = detector.detect(image)
            estimation = estimator.estimate_pose(results)
            # print(estimation)

            gui.overlay_image_observation(image, results)
            io.publish_estimations(estimation, frame_timestamp)

            cv2.imshow("capture", image)
        else:
            print("did not get image")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
