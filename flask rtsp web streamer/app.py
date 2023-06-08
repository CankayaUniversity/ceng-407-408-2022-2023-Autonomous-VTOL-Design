import cv2
from flask import Flask, Response
import time


app = Flask(__name__)


class ObjectDetection:
    def __init__(self):
        self.CONFIDENCE_THRESHOLD = 0.2
        self.NMS_THRESHOLD = 0.4
        self.COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

        self.class_names = []
        with open("classes.txt", "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        self.net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    def detectObj(self, frame):
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

        for (classid, score, box) in zip(classes, scores, boxes):
            if isinstance(classid, int):
                classid = [classid]
            elif isinstance(classid, list):
                classid = classid[0]

            color = self.COLORS[int(classid) % len(self.COLORS)]
            label = "%s : %f" % (self.class_names[classid], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame


class VideoStreaming:
    def __init__(self):
        self.VIDEO = cv2.VideoCapture(0)
        self.flipH = False
        self.detect = False

    def __del__(self):
        self.VIDEO.release()

    def get_frame(self):
        while True:
            ret, frame = self.VIDEO.read()
            if self.flipH:
                frame = cv2.flip(frame, 1)

            if self.detect:
                processed_frame = object_detection.detectObj(frame)
                ret, jpeg = cv2.imencode('.jpg', processed_frame)
            else:
                ret, jpeg = cv2.imencode('.jpg', frame)

            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def toggle_detection(self):
        self.detect = not self.detect

object_detection = ObjectDetection()
video_stream = VideoStreaming()


@app.route('/')
def toggle_detection():
    video_stream.toggle_detection()
    return Response(video_stream.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


def create_app():
    global object_detection, video_stream
    object_detection = ObjectDetection()
    video_stream = VideoStreaming()
    return app


if __name__ == '__main__':
    app.run(host='192.168.1.17')
