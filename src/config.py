import json
import math
import numpy
import numpy.typing

from dataclasses import dataclass
from wpimath.geometry import *

@dataclass
class CalibrationInfo:
    resolution: tuple[int, int]
    matrix: numpy.typing.NDArray[numpy.float64]
    distortion_coeffs: numpy.typing.NDArray[numpy.float64]

@dataclass
class CameraSettings:
    id: int
    name: str
    calibration: CalibrationInfo

@dataclass
class TagEnvironment:
    tag_size: float
    tags: dict[int, Pose3d]

    def get_tag_pose(self, id: int) -> Pose3d:
        if id not in self.tags:
            return None
        return self.tags[id]
    
@dataclass
class NetworkTablesConfig:
    server_ip: str
    identity: str

@dataclass
class StreamConfig:
    port: int

@dataclass
class LoggingConfig:
    enabled: bool
    output_dir: str

@dataclass
class TagTrackerConfig:
    environment: TagEnvironment
    networktables: NetworkTablesConfig
    tag_family: str
    process_threads: int
    cameras: list[CameraSettings]
    stream: StreamConfig
    logging: LoggingConfig

def load_calibration(file_name: str) -> CalibrationInfo:
    with open(file_name, 'r') as json_file:
        json_obj = json.load(json_file)

    res_obj = json_obj["camera_resolution"]
    res_data = res_obj["data"]
    res = (res_data[1], res_data[0])

    mtx_obj = json_obj["camera_matrix"]
    mtx = numpy.array(mtx_obj["data"]).reshape(3, 3)

    dist_obj = json_obj["distortion_coefficients"]
    dist = numpy.array(dist_obj["data"])

    return CalibrationInfo(
        resolution=res,
        matrix=mtx,
        distortion_coeffs=dist
    )

def load_environment(file_name: str) -> TagEnvironment:
    with open(file_name, 'r') as json_file:
        json_obj = json.load(json_file)

    tags = {}
    tag_size = json_obj["tag_size"]
    json_tags = json_obj["tags"]
    for json_tag in json_tags:
        id = json_tag["ID"]

        json_pose = json_tag["pose"]
        json_translation = json_pose["translation"]
        tx = json_translation["x"]
        ty = json_translation["y"]
        tz = json_translation["z"]

        json_rotation = json_pose["rotation"]

        if "euler" in json_rotation:
            json_euler = json_rotation["euler"]
            rx = math.radians(json_euler["X"])
            ry = math.radians(json_euler["Y"])
            rz = math.radians(json_euler["Z"])
            rotation = Rotation3d(rx, ry, rz)
        else:
            json_quat = json_rotation["quaternion"]
            qw = json_quat["W"]
            qx = json_quat["X"]
            qy = json_quat["Y"]
            qz = json_quat["Z"]
            rotation = Rotation3d(Quaternion(qw, qx, qy, qz))

        pose = Pose3d(
            Translation3d(tx, ty, tz),
            rotation
        )

        tags[id] = pose

    return TagEnvironment(
        tag_size=tag_size,
        tags=tags
    )
    
def load_config(file_name: str) -> TagTrackerConfig:
    with open(file_name, 'r') as json_file:
        json_obj = json.load(json_file)

    nt_obj = json_obj["networktables"]
    stream_obj = json_obj["web-stream"]
    logging_obj = json_obj["logging"]

    cameras = []
    for camera_obj in json_obj["cameras"]:
        cameras.append(CameraSettings(
            id=camera_obj["id"],
            name=camera_obj["name"],
            calibration=load_calibration(camera_obj["calibration"])
        ))

    return TagTrackerConfig(
        environment=load_environment(json_obj["environment"]),
        networktables=NetworkTablesConfig(
            server_ip=nt_obj["server-ip"],
            identity=nt_obj["identity"]
        ),
        tag_family=json_obj["tag-family"],
        process_threads=json_obj["process-threads"],
        cameras=cameras,
        stream=StreamConfig(
            port=stream_obj["port"]
        ),
        logging=LoggingConfig(
            enabled=logging_obj["enabled"],
            output_dir=logging_obj["output-dir"]
        )
    )
