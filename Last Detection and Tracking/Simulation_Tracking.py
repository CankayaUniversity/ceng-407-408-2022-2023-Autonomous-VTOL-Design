import cv2
import numpy as np
import math
from math import sqrt
import datetime
import imutils
import asyncio
from mavsdk import System
from mavsdk.offboard import PositionNedYaw
from mavsdk.offboard import OffboardError


PERCENT_CENTER_RECT = 0.20  # For calculating the center rectangle's size
PERCENT_TARGET_RADIUS = 0.25 * PERCENT_CENTER_RECT  # Minimum target radius to follow
HOVERING_ALTITUDE = 15.0  # Altitude in meters to which the drone will perform its tasks
NUM_FILT_POINTS = 20  # Number of filtering points for the Moving Average Filter
DESIRED_IMAGE_HEIGHT = 480  # A smaller image makes the detection less CPU intensive

# A dictionary of two empty buffers (arrays) for the Moving Average Filter
filt_buffer = {'width': [], 'height': []}

params = {'image_height': None, 'image_width': None, 'resized_height': None, 'resized_width': None,
          'x_ax_pos': None, 'y_ax_pos': None, 'cent_rect_half_width': None, 'cent_rect_half_height': None,
          'cent_rect_p1': None, 'cent_rect_p2': None, 'scaling_factor': None, 'min_tgt_radius': None}

def find_objects(outputs, img, classNames, confThreshold, nmsThreshold):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]

        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 50, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 50, 255), 2)

        return x, y, w, h


async def run():
    #cap = cv2.VideoCapture("Test_Video/5.mp4")
    cap = cv2.VideoCapture("http://10.100.192.88:5000/video_feed")

    if cap.isOpened() is False:
        print('[ERROR] couldnt open the camera.')
        return

    print('camera opened successfully')

    await get_image_params(cap)
    print(f"-- Original image width, height: {params['image_width']}, {params['image_height']}")

    drone = System()
    await drone.connect(system_address="udp://:14540") #for simulation
    # await drone.connect(system_address="serial:///dev/ttyUSB0:57600") telemetry module connected drone

    # Asynchronously poll the connection state until receiving an 'is_connected' confirmation
    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"Drone discovered with UUID: ")
            break

    # Activate the drone motors
    print("-- Arming")
    await drone.action.arm()

    # Send an initial position to the drone before changing to "offboard" flight mode
    print("-- Setting initial setpoint")
    await drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))

    # Change flight mode to "offboard", if it fails, disarm the motors and abort the script
    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return

    # Variables to store the NED coordinates, plus Yaw angle. Default pose:
    N_coord = 0
    E_coord = 0
    D_coord = -HOVERING_ALTITUDE  # The drone will always detect and track at HOVERING_ALTITUDE
    yaw_angle = 0  # Drone always points to North

    # Make the drone go (take off) to the default pose
    await drone.offboard.set_position_ned(
        PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))

    await asyncio.sleep(4)  # Give the drone time to gain altitude

    classesFile = "classes.txt"

    a = 0
    distance = 0
    whT = 320
    classNames = []
    confThreshold =  0.2
    nmsThreshold = 0.4

    control_distance=10

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')

    net = cv2.dnn.readNet("custom-yolov4-tiny-detector_best.weights", "custom-yolov4-tiny-detector.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

    while True:
        success, img = cap.read()

        img = cv2.flip(img, 1)

        gray_video = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
        cv2.rectangle(img, (int(img.shape[1] / 4), int(img.shape[0] / 10)),
                      (int(3 * img.shape[1] / 4), int(9 * img.shape[0] / 10)), (204, 0, 102), 3)
        cv2.rectangle(img, (0, 0), (int(img.shape[1]), int(img.shape[0])), (0, 153, 76), 4)
        # cv2.circle(img,(int (img.shape[1]/2),int (img.shape[0]/2)))

        cv2.putText(img, "FOV : Field of View", (10, int(img.shape[0]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 0),
                    2, cv2.LINE_AA, False)
        cv2.putText(img, "Target : Target of View", (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA, False)

        outputs = net.forward(outputNames)
        find_objects(outputs, img, classNames, confThreshold, nmsThreshold)

        returner = find_objects(outputs, img, classNames, confThreshold, nmsThreshold)

        print(returner)

        if returner is not None:
            cv2.line(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)),
                     (int(returner[0] + returner[2] / 2), int(returner[1] + returner[3] / 2)), (0, 255, 0), 2)
            roi_img = img[returner[1]:returner[1] + returner[3], returner[0]:returner[0] + returner[2]]

            distance = ((returner[0] - int(img.shape[0] / 2)) ** 2 + (returner[1] - int(img.shape[0] / 2) ** 2))
            cv2.putText(img, "Distance:" + ('%d' % int(distance)), (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200),2, cv2.LINE_AA, False)

            horizantal_difference = int(returner[0] + returner[2] / 2) - int(img.shape[1] / 2)
            if horizantal_difference > 0:
                print("right")
                cv2.putText(img, "Right:"+ ('%.2f' % float((horizantal_difference)/control_distance)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
                E_coord = ((horizantal_difference)/control_distance)
                print((horizantal_difference)/control_distance)
            else:
                print("left")
                cv2.putText(img, "Left:"+ ('%.2f' % float((horizantal_difference*-1)/control_distance)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
                E_coord = ((horizantal_difference)/control_distance)
                print("left:",(horizantal_difference*-1)/control_distance)

            vertical_difference = int(returner[1] + returner[3] / 2) - int(img.shape[0] / 2)

            if vertical_difference > 0:
                print("down")
                cv2.putText(img, "Down:"+ ('%.2f' % float((vertical_difference)/control_distance)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
                N_coord = -((vertical_difference)/control_distance)

            else:
                print("up")
                cv2.putText(img, "Up:"+ ('%.2f' % float((vertical_difference*-1)/control_distance)), (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA, False)
                N_coord = -((vertical_difference)/control_distance)

            if int(img.shape[1] / 4) < returner[0] < int(3 * img.shape[1] / 4) and int(img.shape[0] / 10) < returner[1] < int(9 * img.shape[0] / 10):
               print("içerde")
               cv2.putText(img, "Inside",
                           (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 220, 0), 2, cv2.LINE_AA, False)
            else:
                print("dışarda")
                cv2.putText(img, "Outside",
                            (int(img.shape[1] / 4) + 5, int(9 * img.shape[0] / 10) - 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 200), 2, cv2.LINE_AA, False)
            print(distance)

            cv2.circle(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), 3, (0, 0, 0), 2)
            cv2.circle(img, (int(returner[0] + returner[2] / 2), int(returner[1] + returner[3] / 2)), 3, (0, 0, 255), 2)

            await drone.offboard.set_position_ned(
                PositionNedYaw(N_coord, E_coord, D_coord, yaw_angle))


        cv2.imshow('Image', img)

        cv2.waitKey(1)

async def get_image_params(vid_cam):
    # Grab a frame and get its size
    is_grabbed, frame = vid_cam.read()
    params['image_height'], params['image_width'], _ = frame.shape

    # Compute the scaling factor to scale the image to a desired size
    if params['image_height'] != DESIRED_IMAGE_HEIGHT:
        params['scaling_factor'] = round((DESIRED_IMAGE_HEIGHT / params['image_height']), 2)  # Rounded scaling factor

    else:
        params['scaling_factor'] = 1

    print("params['scaling_factor']: ", params['scaling_factor'])

    # Compute resized width and height and resize the image
    params['resized_width'] = int(params['image_width'] * params['scaling_factor'])
    params['resized_height'] = int(params['image_height'] * params['scaling_factor'])
    dimension = (params['resized_width'], params['resized_height'])
    frame = cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)

    # Compute the center rectangle's half width and half height
    params['cent_rect_half_width'] = round(
        params['resized_width'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent (0.5)
    params['cent_rect_half_height'] = round(
        params['resized_height'] * (0.5 * PERCENT_CENTER_RECT))  # Use half percent (0.5)

    # Compute the minimum target radius to follow. Smaller detected targets will be ignored
    params['min_tgt_radius'] = round(params['resized_width'] * PERCENT_TARGET_RADIUS)

    # Compute the position for the X and Y Cartesian coordinates in camera pixel units
    params['x_ax_pos'] = int(params['resized_height'] / 2 - 1)
    params['y_ax_pos'] = int(params['resized_width'] / 2 - 1)

    # Compute two points: p1 in the upper left and p2 in the lower right that will be used to
    # draw the center rectangle iin the image frame
    params['cent_rect_p1'] = (params['y_ax_pos'] - params['cent_rect_half_width'],
                              params['x_ax_pos'] - params['cent_rect_half_height'])
    params['cent_rect_p2'] = (params['y_ax_pos'] + params['cent_rect_half_width'],
                              params['x_ax_pos'] + params['cent_rect_half_height'])

    return




if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())

