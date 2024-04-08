# This is inteneded to be ran through SSH and operates on its own
import cv2
import numpy as np

import warnings
import os
import datetime

from typing import List

CALIBRATION_FILENAME = "calibration.json"
DOWNSCALE = 0.4 # For fitting on screen

def main():
    all_corners: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    imsize = None
    coverage_map = None

    # Create detecor and representation of ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    #square_size_m = 0.13592 / 6
    square_size_m = 0.2579 / 12
    charuco_board = cv2.aruco.CharucoBoard((12,9), square_size_m, square_size_m * 7/9, aruco_dict) # Different from 6328
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board, cv2.aruco.CharucoParameters(), aruco_params)

    # Create webcam
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

    # Configure camera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 50)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    while True:
        ret, color = cap.read()
        if not ret:
            print("Capture failed")
            break

        # Convert to grayscale
        image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if imsize is None:
            imsize = (image.shape[0], image.shape[1])

            # Initialize empty image
            coverage_map = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

        # Detect tags
        corners, ids, rejected = detector.detectMarkers(image)
        charuco_corners = None
        charuco_ids = None
        can_save = False
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find ChArUco
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, charuco_board)

            if ret > 20:
                can_save = True
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

        small = cv2.resize(image, (0, 0), fx = DOWNSCALE, fy = DOWNSCALE)
        if can_save:
            cv2.putText(small, "Frame good!", (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (64, 255, 64), 2)
        cv2.imshow("Image", small)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and can_save:
            # Save the corners
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            print("Saved frame for calibration")

            # Plot all the stuff on the coverage map so you can see where we've calibrated
            # TODO: Plot the region of the board instead so it's clearer
            cv2.aruco.drawDetectedMarkers(coverage_map, corners)
            cv2.aruco.drawDetectedCornersCharuco(coverage_map, charuco_corners, charuco_ids)

            cv2.imshow("Last frame", small)
            cv2.imshow("Coverage", cv2.resize(coverage_map, (0, 0), fx = DOWNSCALE, fy = DOWNSCALE))

    print("Calibration may take some time, be patient")

    # Check if nothing even got seen
    if len(all_corners) == 0:
        print("No calibration images found")
        quit()
    
    # Delete the previous calibration
    if os.path.exists(CALIBRATION_FILENAME):
        os.remove(CALIBRATION_FILENAME)

    reproj_err, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners, charucoIds=all_ids, board=charuco_board, imageSize=imsize, cameraMatrix=None, distCoeffs=None
    )

    storage = cv2.FileStorage(CALIBRATION_FILENAME, cv2.FILE_STORAGE_WRITE)
    storage.write("calibration_date", str(datetime.datetime.now()))
    storage.write("camera_resolution", imsize)
    storage.write("camera_matrix", camera_matrix)
    storage.write("distortion_coefficients", distortion_coefficients)
    storage.write("reprojection_err", reproj_err)
    storage.release()
    print("Finished calibration")
    print("Reprojection error:", reproj_err)
    
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
    exit()

    all_corners: List[np.ndarray] = []
    all_ids: List[np.ndarray] = []
    imsize = None

    # Create detecor and representation of ChArUco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    #square_size_m = 0.13592 / 6
    square_size_m = 0.2579 / 12
    charuco_board = cv2.aruco.CharucoBoard((12,9), square_size_m, square_size_m * 7/9, aruco_dict) # Different from 6328
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board, cv2.aruco.CharucoParameters(), aruco_params)

    # Create webcam
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)

    # Try to set resolution to really big, OpenCV will fall back to
    # the camera's highest supported resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FPS, 50)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    frame_count = 0
    while True:
        ret, color = cap.read()

        frame_count += 1
        if (frame_count % 10 != 1):
            continue

        # Convert to grayscale
        image = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

        if imsize is None:
            imsize = (image.shape[0], image.shape[1])

        # Detect tags
        corners, ids, rejected = detector.detectMarkers(image)

        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find ChArUco
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, image, charuco_board)

            if ret > 20:

                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

                # Save the corners
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)
                print("Saved frame for calibration")

        # cv2.aruco.drawDetectedCornersCharuco(image, corners)
        small = cv2.resize(image, (0, 0), fx = 0.5, fy = 0.5)
        cv2.imshow("Image", small)
        #cv2.imshow("Image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Calibration may take some time, be patient")

    # Check if nothing even got seen
    if len(all_corners) == 0:
        print("No calibration images found")
        quit()
    
    # Delete the previous calibration
    if os.path.exists(CALIBRATION_FILENAME):
        os.remove(CALIBRATION_FILENAME)

    ret, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_corners, charucoIds=all_ids, board=charuco_board, imageSize=imsize, cameraMatrix=None, distCoeffs=None
    )

    if ret:
        storage = cv2.FileStorage(CALIBRATION_FILENAME, cv2.FILE_STORAGE_WRITE)
        storage.write("calibration_date", str(datetime.datetime.now()))
        storage.write("camera_resolution", imsize)
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", distortion_coefficients)
        storage.release()
        print("Finished calibration")
    else:
        print("Something went wrong when calculating the calibration")
    
    cv2.destroyAllWindows()
    cap.release()

        
else:
    warnings.warn("This file should be ran through the command prompt, not as a package")
