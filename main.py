import flask
from flask_cors import CORS
from flask import request, jsonify, send_from_directory
import logging
import base64
from PIL import Image
import io
from io import StringIO
import numpy as np
import argparse
import time
import cv2
import os
import classifier


app = flask.Flask("__main__")
CORS(app)


def indentifyCars(base64_string_data):
    ap = argparse.ArgumentParser()
    ap.add_argument("-y", "--yolo", default='yolo-coco',
                    help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.3,
                    help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())
    car_color_classifier = classifier.Classifier()

    # load the COCO class labels our YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
    configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    # load our input image and grab its spatial dimensions
    def readb64(base64_string):
        sbuf = io.BytesIO()
        sbuf.write(base64.b64decode(base64_string))
        pimg = Image.open(sbuf)
        return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

    image = readb64(base64_string_data)
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    outputs = net.forward(output_layers)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                            args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            if classIDs[i] == 2:
                start = time.time()
                result = car_color_classifier.predict(
                    image[max(y, 0):y + h, max(x, 0):x + w])
                end = time.time()
                # show timing information on MobileNet classifier
                print("[INFO] classifier took {:.6f} seconds".format(
                    end - start))
                text = "{}: {:.4f}".format(
                    result[0]['make'], float(result[0]['prob']))
                cv2.putText(image, text, (x + 2, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
                cv2.putText(image, result[0]['model'], (x + 2, y + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    # show the output image
    cv2.imwrite("output.jpg", image)
    # get base64_string of image output
    with open("output.jpg", "rb") as image_file:
        data = base64.b64encode(image_file.read())

    return data


@app.route('/result', methods=['POST'])
def result():
    logging.basicConfig(level=logging.DEBUG)
    base64_string = request.json
    if base64_string:
        base64_encoded = indentifyCars(base64_string)
        logging.warning(base64_encoded.decode('utf-8'))
        return jsonify(base64_encoded.decode('utf-8'))
    return "No information is given"


@app.route("/")
def my_index():
    return flask.render_template("index.html", token="Hello from Danh")


app.run(debug=True)
