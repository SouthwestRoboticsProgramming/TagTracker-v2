import math
import ntcore
from dataclasses import dataclass
from wpimath.geometry import *

import config
import process

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

def append_estimate(pose_data: list[float], est: tuple[Pose3d, float]):
    pose, err = est

    pose_data.append(err)
    append_pose(pose_data, pose)

class CameraNetworkTablesIO:
    def __init__(self, camera: str):
        inst = ntcore.NetworkTableInstance.getDefault()
        table = inst.getTable("/TagTracker/Cameras/" + camera)

        self.poses_pub = table.getDoubleArrayTopic("poses").publish(
            ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        self.fps_pub = table.getDoubleTopic("fps").publish()

    def publish_output(self, result: process.FrameResult):
        est = result.estimates

        # Collect the pose data into an array
        pose_data = [0]
        if est:
            pose_data[0] = 1
            append_estimate(pose_data, est.pose_a)
            if est.pose_b is not None:
                pose_data[0] = 2
                append_estimate(pose_data, est.pose_b)
            for detection in result.detections:
                pose_data.append(detection.id)

        # Send it
        # Set the entry's timestamp so robot code knows when frame was captured
        self.poses_pub.set(pose_data, math.floor(result.frame.timestamp * 10e5))
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

    def publish_output(self, result: process.FrameResult):
        frame = result.frame
        if not frame.camera in self.cameras:
            io = CameraNetworkTablesIO(frame.camera)
            self.cameras[frame.camera] = io
        else:
            io = self.cameras[frame.camera]
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
