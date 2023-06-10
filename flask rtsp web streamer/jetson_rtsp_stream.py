import time
from flask import Response, Flask
from threading import Thread, Lock
import cv2


global video_frame
video_frame = None

global thread_lock
thread_lock = Lock()


class CSICamera:

    def gstreamer_pipeline(self, capture_width=3280, capture_height=2464, output_width=224, output_height=224,
                           framerate=21, flip_method=0):
        return 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
            capture_width, capture_height, framerate, flip_method, output_width, output_height)

    def __init__(self, image_w=160, image_h=120, image_d=3, capture_width=3280, capture_height=2464, framerate=60,
                 gstreamer_flip=0):
        '''
        gstreamer_flip = 0 - no flip
        gstreamer_flip = 1 - rotate CCW 90
        gstreamer_flip = 2 - flip vertically
        gstreamer_flip = 3 - rotate CW 90
        '''
        self.w = image_w
        self.h = image_h
        self.running = True
        self.frame = None
        self.flip_method = gstreamer_flip
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.framerate = framerate

    def init_camera(self):
        # initialize the camera and stream
        self.camera = cv2.VideoCapture(
            self.gstreamer_pipeline(
                capture_width=self.capture_width,
                capture_height=self.capture_height,
                output_width=self.w,
                output_height=self.h,
                framerate=self.framerate,
                flip_method=self.flip_method),
            cv2.CAP_GSTREAMER)

        self.poll_camera()
        print('CSICamera loaded... Warming up the camera')
        time.sleep(2)

    def update(self):
        self.init_camera()
        while self.running:
            self.poll_camera()

            # Check for key press
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                self.running = False
                break

        self.shutdown()

    def poll_camera(self):
        global video_frame, thread_lock

        self.ret, frame = self.camera.read()
        if frame is not None:
            with thread_lock:
                video_frame = frame.copy()

            self.frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            cv2.imshow('Frame', self.frame)

    def run(self):
        self.poll_camera()
        return self.frame

    def run_threaded(self):
        return self.frame

    def shutdown(self):
        self.running = False
        print('Stopping CSICamera')
        time.sleep(0.5)
        del self.camera
        cv2.destroyAllWindows()


def encodeFrame():
    global thread_lock

    while True:
        # Acquire thread_lock to access the global video_frame object
        with thread_lock:
            global video_frame
            if video_frame is None:
                continue
            return_key, encoded_image = cv2.imencode(".jpg", video_frame)
            if not return_key:
                continue

        # Output image as a byte array
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encoded_image) + b'\r\n')


# Create the Flask object for the application
app = Flask(__name__)


@app.route("/")
def streamFrames():
    return Response(encodeFrame(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    cam = CSICamera(image_w=224, image_h=224, capture_width=1080, capture_height=720)
    process_thread = Thread(target=cam.update)
    process_thread.daemon = True
    process_thread.start()

    app.run(host='0.0.0.0', threaded=True)

    cam.shutdown()

