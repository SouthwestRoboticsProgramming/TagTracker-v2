import ntcore
import struct
import time
from dataclasses import dataclass
from wpimath.geometry import *

import config
import process
import capture

@dataclass
class MatchInfo:
    event_name: str
    match_num: int
    match_type: int
    replay_num: int
    is_red: bool
    station_num: int

def append_pose(pose_data: list[float], pose: Pose3d):
    tx = pose.translation()
    q = pose.rotation().getQuaternion()

    pose_data.extend([
        tx.X(), tx.Y(), tx.Z(),
        q.W(), q.X(), q.Y(), q.Z()
    ])
    
def pack_estimate(est: tuple[Pose3d, float]) -> bytes:
    pose, err = est

    tx = pose.translation()
    q = pose.rotation().getQuaternion()

    return struct.pack(
        ">fddddddd",
        err,
        tx.X(), tx.Y(), tx.Z(),
        q.W(), q.X(), q.Y(), q.Z()
    )

class CameraNetworkTablesIO:
    def __init__(self, camera: str):
        self.inst = ntcore.NetworkTableInstance.getDefault()
        table = self.inst.getTable("/TagTracker/Cameras/" + camera)

        self.config_table = table.getSubTable("Config")

        output_table = table.getSubTable("Outputs")
        self.poses_pub = output_table.getRawTopic("poses").publish(
            "raw",
            ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        self.fps_pub = output_table.getDoubleTopic("fps").publish()
        self.resolution_pub = output_table.getIntegerArrayTopic("resolution").publish()

    def get_config_params(self) -> capture.CameraParams:
        if not self.inst.isConnected():
            return None

        return capture.CameraParams(
            auto_exposure=self.config_table.getEntry("Auto Exposure").getBoolean(False),
            exposure=self.config_table.getEntry("Exposure").getDouble(42),
            gain=self.config_table.getEntry("Gain").getDouble(1),
            target_fps=self.config_table.getEntry("Target FPS").getDouble(50)
        )

    def publish_image_resolution(self, width: int, height: int):
        self.resolution_pub.set([width, height])

    def publish_output(self, result: process.FrameResult):
        est = result.estimates

        if est:
            pose_data = struct.pack(">?", True)
            pose_data += pack_estimate(est.pose_a)
            if est.pose_b is not None:
                pose_data += struct.pack(">?", True)
                pose_data += pack_estimate(est.pose_b)
                pose_data += struct.pack(">B", len(result.detections))
            else:
                pose_data += struct.pack(">?", False)
                # Length not needed, guaranteed to be exactly one tag
            for detection in result.detections:
                corners = detection.corners
                pose_data += struct.pack(
                    ">BHHHHHHHH",
                    detection.id,

                    # Send the corners for visualization in AdvantageScope
                    round(corners[0][0]), round(corners[0][1]),
                    round(corners[1][0]), round(corners[1][1]),
                    round(corners[2][0]), round(corners[2][1]),
                    round(corners[3][0]), round(corners[3][1])
                )
        else:
            pose_data = struct.pack(">?", False)

        # Send it
        # Send how long it took to process the frame for latency correction
        pose_data += struct.pack(">d", time.monotonic() - result.frame.timestamp)
        self.poses_pub.set(pose_data)
        self.fps_pub.set(result.frame.rate)

class NetworkTablesIO:
    cameras: dict[str, CameraNetworkTablesIO]
    fms: ntcore.NetworkTable

    def __init__(self, conf: config.NetworkTablesConfig, env: config.TagEnvironment):
        self.cameras = {}

        nt = ntcore.NetworkTableInstance.getDefault()
        nt.setServer(conf.server_ip)
        nt.startClient4(conf.identity)

        # Publish the tag environment so ShuffleLog can visualize it
        self.env_pub = nt.getDoubleArrayTopic("/TagTracker/environment").publish()
        env_data = []
        for id, pose in env.tags.items():
            env_data.append(id)
            append_pose(env_data, pose)
        self.env_pub.set(env_data)

        self.fms = nt.getTable("FMSInfo")

    def get_camera_io(self, cam_name: str) -> CameraNetworkTablesIO:
        if not cam_name in self.cameras:
            io = CameraNetworkTablesIO(cam_name)
            self.cameras[cam_name] = io
            return io
        else:
            return self.cameras[cam_name]

    def publish_output(self, result: process.FrameResult):
        frame = result.frame
        io = self.get_camera_io(frame.camera)
        io.publish_output(result)

    def get_match_info(self) -> MatchInfo:
        return MatchInfo(
            event_name=self.fms.getString("EventName", "UNKNOWN"),
            match_num=self.fms.getEntry("MatchNumber").getInteger(999999),
            match_type=self.fms.getEntry("MatchType").getInteger(999999),
            replay_num=self.fms.getEntry("ReplayNumber").getInteger(999999),
            is_red=self.fms.getBoolean("IsRedAlliance", False),
            station_num=self.fms.getEntry("StationNumber").getInteger(999999)
        )
