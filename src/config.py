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
class FrameDebugConfig:
    enabled: bool
    output_dir: str

@dataclass
class StreamConfig:
    port: int

@dataclass
class LoggingConfig:
    enabled: bool
    output_dir: str

@dataclass
class TagTrackerConfig:
    networktables: NetworkTablesConfig
    tag_family: str
    process_threads: int
    cameras: list[CameraSettings]
    frame_debug: FrameDebugConfig
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
    
def load_config(file_name: str) -> TagTrackerConfig:
    with open(file_name, 'r') as json_file:
        json_obj = json.load(json_file)

    nt_obj = json_obj["networktables"]
    frame_debug_obj = json_obj["frame-debug"]
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
        networktables=NetworkTablesConfig(
            server_ip=nt_obj["server-ip"],
            identity=nt_obj["identity"]
        ),
        tag_family=json_obj["tag-family"],
        process_threads=json_obj["process-threads"],
        cameras=cameras,
        frame_debug=FrameDebugConfig(
            enabled=frame_debug_obj["enabled"],
            output_dir=frame_debug_obj["output-dir"]
        ),
        stream=StreamConfig(
            port=stream_obj["port"]
        ),
        logging=LoggingConfig(
            enabled=logging_obj["enabled"],
            output_dir=logging_obj["output-dir"]
        )
    )
