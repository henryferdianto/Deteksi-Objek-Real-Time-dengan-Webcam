#!/usr/bin/env python3
"""
Real-time object detection using YOLOv3-tiny and OpenCV.

Usage:
    python detect.py --weights yolo/yolov3-tiny.weights --cfg yolo/yolov3-tiny.cfg --names yolo/coco.names

Defaults assume files are in ./yolo (the notebook can download them for you).
"""
import os
import time
import argparse
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description='Real-time YOLOv3-tiny object detection')
    p.add_argument('--weights', default='yolo/yolov3-tiny.weights')
    p.add_argument('--cfg', default='yolo/yolov3-tiny.cfg')
    p.add_argument('--names', default='yolo/coco.names')
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--conf', type=float, default=0.5)
    p.add_argument('--nms', type=float, default=0.4)
    return p.parse_args()


def load_yolo(cfg_path, weights_path, names_path):
    if not os.path.exists(cfg_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
        raise FileNotFoundError('One or more YOLO files not found. Check paths for cfg, weights and names.')

    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    # Use OpenCV backend and CPU target by default
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Get output layer names
    try:
        ln = net.getUnconnectedOutLayersNames()
    except AttributeError:
        # Fallback for older OpenCV
        ln = [layer[0] for layer in net.getUnconnectedOutLayers()]

    with open(names_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    return net, ln, classes


def detect_loop(net, ln, classes, camera_index=0, conf_threshold=0.5, nms_threshold=0.4):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print('Cannot open camera. Check the camera index and permissions.')
        return

    (W, H) = (None, None)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to grab frame')
                break

            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # Create blob and perform a forward pass
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            start = time.time()
            layerOutputs = net.forward(ln)
            end = time.time()

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = int(np.argmax(scores))
                    confidence = float(scores[classID])
                    if confidence > conf_threshold:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype('int')
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f"{classes[classIDs[i]]}: {confidences[i]:.2f}"
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            fps_text = f"FPS: {1/(end-start):.2f}"
            cv2.putText(frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('YOLO Real-time', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    net, ln, classes = load_yolo(args.cfg, args.weights, args.names)
    detect_loop(net, ln, classes, camera_index=args.camera, conf_threshold=args.conf, nms_threshold=args.nms)


if __name__ == '__main__':
    main()
