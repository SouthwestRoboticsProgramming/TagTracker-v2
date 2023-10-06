import pipeline
import cv2

def main():
    settings = pipeline.CameraSettings(
        opencv_id=0,
        resolution=[640, 480],
        auto_exposure=1, # Theoretically this should turn it off
        exposure=100,
        gain=10
    )
    capture = pipeline.Capture(settings)

    while True:
        retval, image = capture.read_frame()
        if retval:
            cv2.imshow("capture", image)
        else:
            print("did not get image")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
