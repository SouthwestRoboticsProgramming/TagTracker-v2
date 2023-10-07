from wpimath.geometry import *
import json

class TagEnvironment:
    tag_size: float
    tags: dict[int, Pose3d]

    def __init__(self, json_file_name):
        json_file = open(json_file_name, 'r')
        json_obj = json.load(json_file)
        
        self.tags = {}
        self.tag_size = json_obj["tag_size"]
        json_tags = json_obj["tags"]
        for json_tag in json_tags:
            id = json_tag["ID"]

            json_pose = json_tag["pose"]
            json_translation = json_pose["translation"]
            tx = json_translation["x"]
            ty = json_translation["y"]
            tz = json_translation["z"]

            json_rotation = json_pose["rotation"]
            json_quat = json_rotation["quaternion"]
            qw = json_quat["W"]
            qx = json_quat["X"]
            qy = json_quat["Y"]
            qz = json_quat["Z"]

            pose = Pose3d(
                Translation3d(tx, ty, tz),
                Rotation3d(Quaternion(qw, qx, qy, qz))
            )

            self.tags[id] = pose
    
    def get_tag_pose(self, id: int) -> Pose3d:
        if id not in self.tags:
            return None
        return self.tags[id]