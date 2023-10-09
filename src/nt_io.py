import ntcore
import solve
import math
from wpimath.geometry import *

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

class NetworkTablesIO:
    def __init__(self, camera_name: str):
        # Get NT publisher
        inst = ntcore.NetworkTableInstance.getDefault()
        table = inst.getTable("/TagTracker/Cameras/" + camera_name)
        self.poses_pub = table.getDoubleArrayTopic("poses").publish(
            ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
        self.fps_pub = table.getDoubleTopic("fps").publish()

    def publish_data(self, estimation: solve.PoseEstimation, timestamp: float, fps: float) -> None:
        # Collect the pose data into an array
        pose_data = [0]
        if estimation:
            pose_data[0] = 1
            append_estimate(pose_data, estimation.estimate_a)
            if estimation.estimate_b:
                pose_data[0] = 2
                append_estimate(pose_data, estimation.estimate_b)
            for tag_id in estimation.ids:
                pose_data.append(tag_id)

        # Send it
        # Set the entry's timestamp so robot code knows when frame was captured
        self.poses_pub.set(pose_data, math.floor(timestamp * 10e5))
        self.fps_pub.set(fps)
