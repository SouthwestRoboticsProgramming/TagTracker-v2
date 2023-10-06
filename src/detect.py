import cv2

class TagDetector:
    def __init__(self, dictionary):
        dict = cv2.aruco.getPredefinedDictionary(dictionary)
        params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dict, params)

    def detect(self, image: cv2.Mat):
        corners, ids, _ = self.detector.detectMarkers(image)
        if len(corners) == 0:
            return []
        print("corners", corners, " ids", ids)
