import numpy
import numpy.typing
import cv2
import detect
import geometry
import math
from dataclasses import dataclass

@dataclass
class PoseEstimation:
    id: int
    # Tuple of (pose, error)
    estimate_a: tuple[geometry.Pose3d, float]
    estimate_b: tuple[geometry.Pose3d, float]

@dataclass
class CalibrationInfo:
    matrix: numpy.typing.NDArray[numpy.float64]
    distortion_coeffs: numpy.typing.NDArray[numpy.float64]

def calcPose(tvec: numpy.typing.NDArray[numpy.float64], rvec: numpy.typing.NDArray[numpy.float64]) -> geometry.Pose3d:
    # Different from 6328 (they swapped and inverted the vector components)
    return geometry.Pose3d(
        translation=(tvec[0][0], tvec[1][0], tvec[2][0]),
        rotation_axis=(rvec[0][0], rvec[1][0], rvec[2][0]),
        rotation_angle_rad=math.sqrt(math.pow(rvec[0][0], 2) + math.pow(rvec[1][0], 2) + math.pow(rvec[2][0], 2))
    )

class PoseEstimator:
    def __init__(self, tag_size, calibration: CalibrationInfo):
        self.tag_size = tag_size
        self.calibration = calibration

    def estimate_pose(self, tag_info: detect.DetectedTag) -> PoseEstimation:
        half_sz = self.tag_size / 2.0
        object_points = numpy.array([[-half_sz, half_sz, 0.0],
                                     [half_sz, half_sz, 0.0],
                                     [half_sz, -half_sz, 0.0],
                                     [-half_sz, -half_sz, 0.0]])
        
        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points, tag_info.corners, self.calibration.matrix,
                self.calibration.distortion_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        except Exception as e:
            print(e)
            return None
        
        print(rvecs, tvecs, errors)
        
        return PoseEstimation(
            id=tag_info.id,
            estimate_a=(calcPose(tvecs[0], rvecs[0]), errors[0][0]),
            estimate_b=(calcPose(tvecs[1], rvecs[1]), errors[1][0])
        )
