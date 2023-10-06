import ntcore
import solve

class NetworkTablesIO:
    def __init__(self, server_ip, client_name):
        inst = ntcore.NetworkTableInstance.getDefault()
        inst.setServer(server_ip)
        inst.startClient4(client_name)

        table = inst.getTable("/TagTracker/" + client_name + "/outputs")
        self._poses_pub = table.getDoubleArrayTopic("poses")
    
    def publish_estimations(estimations: list[solve.PoseEstimation]) -> None:
        pass
