from dataclasses import dataclass

@dataclass
class Pose3d:
    translation: tuple[float, float, float]
    rotation_axis: tuple[float, float, float]
    rotation_angle_rad: float
