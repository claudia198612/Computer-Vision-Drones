# coding=utf-8
import numpy as np
import os

PROTOTXT_PATH = os.path.join('..', 'models', 'MobileNetSSD_deploy.prototxt.txt')
MODEL_PATH = os.path.join('..', 'models', 'MobileNetSSD_deploy.caffemodel')

DEFAULT_CONFIDENCE = 0.5
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS[6] = (255, 255, 0)  # bus
COLORS[7] = (255, 255, 0)  # car
COLORS[14] = (255, 255, 0)  # motorbike
COLORS[15] = (0, 255, 0)  # person

RIGHT = [3, 0, 0, 50]
LEFT = [-3, 0, 0, 50]
FOLLOW = [0, 3, 0, 50]
BACK = [0, -3, 0, 50]
DEBUG = [0, 0, 0, 20]
