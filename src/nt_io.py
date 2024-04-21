import math
import ntcore
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

def append_estimate(pose_data: list[float], est: tuple[Pose3d, float]):
    pose, err = est

    pose_data.append(err)
    append_pose(pose_data, pose)

class CameraNetworkTablesIO:
    def __init__(self, camera: str):
        self.inst = ntcore.NetworkTableInstance.getDefault()
        table = self.inst.getTable("/TagTracker/Cameras/" + camera)

        self.config_table = table.getSubTable("Config")

        opt = ntcore.PubSubOptions()
        opt.sendAll = True

        self.exposure_sub = self.config_table.getDoubleTopic("Exposure").subscribe(31.0, opt)

        output_table = table.getSubTable("Outputs")
        self.poses_pub = output_table.getDoubleArrayTopic("poses").publish(
            ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        self.fps_pub = output_table.getDoubleTopic("fps").publish()

    def get_config_params(self) -> capture.CameraParams:
        if not self.inst.isConnected():
            return None

        return capture.CameraParams(
            auto_exposure=self.config_table.getEntry("Auto Exposure").getBoolean(False),
            exposure=self.config_table.getEntry("Exposure").getDouble(42),
            gain=self.config_table.getEntry("Gain").getDouble(1),
            target_fps=self.config_table.getEntry("Target FPS").getDouble(50)
        )

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
        # Send how long it took to process the frame for latency correction
        pose_data.append(time.monotonic() - result.frame.timestamp)
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
