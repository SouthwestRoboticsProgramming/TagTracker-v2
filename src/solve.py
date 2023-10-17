import cv2
import math
import numpy
from dataclasses import dataclass
from wpimath.geometry import *

import config
import detect

@dataclass
class EstimatePair:
    pose_a: tuple[Pose3d, float]
    pose_b: tuple[Pose3d, float]

# OpenCV tvec and rvec (from perspective of camera) are +X right, +Y down, +Z forward 
# WPI coordinates (from perspective of identity pose): +X forward, +Y left, +Z up
# So mapping is (wpi_x, wpi_y, wpi_z) = (cv_z, -cv_x, -cv_y)

# Converts OpenCV tvec and rvec to WPI transform from camera to tag
def cvToWpi(tvec: numpy.typing.NDArray[numpy.float64], rvec: numpy.typing.NDArray[numpy.float64]) -> Pose3d:
    return Pose3d(
        Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
        Rotation3d(
            # rvec encodes axis as its direction and angle in radians as its magnitude
            numpy.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(math.pow(rvec[0][0], 2) + math.pow(rvec[1][0], 2) + math.pow(rvec[2][0], 2))
        ))

# Does the opposite of cvToWpi (but only for translation part of it; it outputs the tvec)
def wpiToCv(tx: Translation3d) -> list[float]:
    return [-tx.Y(), -tx.Z(), tx.X()]

class PoseEstimator:
    env: config.TagEnvironment

    def __init__(self, env: config.TagEnvironment):
        self.env = env

    def solve(self, calibration: config.CalibrationInfo, detections: list[detect.DetectedTag]) -> EstimatePair:
        # Collect the corner positions of all the detected tags in field space
        half_sz = self.env.tag_size / 2.0
        object_points = []
        image_points = []
        tag_ids = []
        tag_poses = []
        for tag_info in detections:
            tag_pose = self.env.get_tag_pose(tag_info.id)
            if tag_pose:
                # Get corner positions in field space
                corner_0 = tag_pose + Transform3d(Translation3d(0, half_sz, -half_sz), Rotation3d())
                corner_1 = tag_pose + Transform3d(Translation3d(0, -half_sz, -half_sz), Rotation3d())
                corner_2 = tag_pose + Transform3d(Translation3d(0, -half_sz, half_sz), Rotation3d())
                corner_3 = tag_pose + Transform3d(Translation3d(0, half_sz, half_sz), Rotation3d())
                
                # Put all the things in the lists
                # Important: the indices all align in the lists

                # Object points are the corner positions in CV camera space
                object_points += [
                    wpiToCv(corner_0.translation()),
                    wpiToCv(corner_1.translation()),
                    wpiToCv(corner_2.translation()),
                    wpiToCv(corner_3.translation())
                ]
                # Image points are the pixel coordinates of the corners of the tags
                image_points += [
                    [tag_info.corners[0][0][0], tag_info.corners[0][0][1]],
                    [tag_info.corners[0][1][0], tag_info.corners[0][1][1]],
                    [tag_info.corners[0][2][0], tag_info.corners[0][2][1]],
                    [tag_info.corners[0][3][0], tag_info.corners[0][3][1]]
                ]
                tag_ids.append(tag_info.id)
                tag_poses.append(tag_pose)
        
        if len(tag_ids) == 1:
            # Use tag local space for corner positions
            object_points = numpy.array([[-half_sz, half_sz, 0.0],
                                         [half_sz, half_sz, 0.0],
                                         [half_sz, -half_sz, 0.0],
                                         [-half_sz, -half_sz, 0.0]])
            try:
                # rvecs and tvecs transform from camera position to (0, 0, 0) in the space object_points is in
                # In this case that is tag space, so they transform camera to tag
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points, numpy.array(image_points),
                    calibration.matrix, calibration.distortion_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except Exception as e:
                print(e)
                return None

            # Calculate WPILib camera poses

            # Calculate transforms from camera to each potential tag pose
            camera_to_tag_pose_0 = cvToWpi(tvecs[0], rvecs[0])
            camera_to_tag_pose_1 = cvToWpi(tvecs[1], rvecs[1])
            camera_to_tag_0 = Transform3d(camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation())
            camera_to_tag_1 = Transform3d(camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation())

            # Combine transforms to find camera pose in field space
            # Camera to tag is inverted so overall transform is
            # field to tag + tag to camera = field to camera
            field_to_tag_pose = tag_poses[0]
            field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())

            # Convert back into Pose3d
            field_to_camera_pose_0 = Pose3d(field_to_camera_0.translation(), field_to_camera_0.rotation())
            field_to_camera_pose_1 = Pose3d(field_to_camera_1.translation(), field_to_camera_1.rotation())

            return EstimatePair(
                pose_a=(field_to_camera_pose_0, errors[0][0]),
                pose_b=(field_to_camera_pose_1, errors[1][0])
            )
        elif len(tag_ids) > 1:
            # Multiple tags were found, solve with all of them at once
            try:
                # object_points are in field space, so this finds camera to field transform
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    numpy.array(object_points), numpy.array(image_points),
                    calibration.matrix, calibration.distortion_coeffs, flags=cv2.SOLVEPNP_SQPNP)
            except Exception as e:
                print(e)
                return None
            
            # WPI-ify and invert so it is field to camera
            camera_to_field_pose = cvToWpi(tvecs[0], rvecs[0])
            camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            field_to_camera = camera_to_field.inverse() # camera to field -> field to camera
            field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())

            return EstimatePair(
                pose_a=(field_to_camera_pose, errors[0][0]),
                # No estimate B here, ambiguity should have already been resolved by having
                # multiple tags to sample, since OpenCV will find the set of possibilities
                # that best match each other
                pose_b=None
            )
        else:
            # No known tags were found, we can't estimate anything this frame
            return None