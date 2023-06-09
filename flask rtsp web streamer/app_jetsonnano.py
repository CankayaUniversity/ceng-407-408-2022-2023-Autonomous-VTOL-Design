gstreamer_pipeline = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! "
    "appsink"
)

# Create a VideoCapture object with the GStreamer pipeline
video_capture = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
[01: 43, 10.06
.2023] Ayberk
Aydemir: https: // maker.pro / nvidia - jetson / tutorial / streaming - real - time - video -
from

-rpi - camera - to - browser - on - jetson - nano -
with-flask
    [01: 59, 10.06
    .2023] Ayberk
    Aydemir:
    import cv2
from flask import Flask, Response
import time

app = Flask(_name_)

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'


def generate_frames():
    video_capture = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)

    if not video_capture.isOpened():
        print("Error: Unable to open camera")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    video_capture.release()


@app.route('/')
def stream_camera():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if _name_ == '_main_':
    app.run(host='192.168.1.21')
