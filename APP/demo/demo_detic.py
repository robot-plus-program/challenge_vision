import copy
import glob
import os
import time
import cv2
import numpy as np
import tqdm
import sys
from functools import reduce

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import matplotlib.pyplot as plt
import inst_seg
sys.path.insert(0, 'third_party/CenterNet2/')

from APP.detector.detic_detector import detic_detector

PATH = os.path.dirname(os.path.abspath(__file__)) + '/inst_seg'

CLASSES = ""
SIZE = [1280, 720]
DEMO_TYPE =  'img'
CONFIG_FILE = '../../inst_seg/inst_seg/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
MODEL = 'Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'

CCcount = 0


def prediction_to_array(self, outputs):
    predictions = outputs["instances"].to("cpu")

    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    if boxes:
        boxes = boxes.tensor.numpy()
        scores = scores.numpy()
        classes = classes.numpy()
    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks).astype(np.uint8)
    else:
        masks = None
    return boxes, scores, classes, masks
def main():
    detector=detic_detector()

    rgb_data = cv2.imread("../data/2024_06_19_14_50_37__rgb.png")
    img=copy.deepcopy(rgb_data)

    image,boxes, scores, classes, masks = detector.run(img)
    cv2.imshow("ret",image)
    cv2.waitKey(1000)
    print(boxes, scores, classes, masks)
if __name__ == "__main__":
    main()


