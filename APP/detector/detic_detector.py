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
from centernet.config import add_centernet_config
from inst_seg.detic.config import add_detic_config

from inst_seg.detic.predictor import Predictor
from inst_seg.utils.cfg_namespace import AlgorithmConfig


PATH = os.path.dirname(os.path.abspath(__file__)) + '/inst_seg'

CLASSES = ""
SIZE = [1280, 720]
DEMO_TYPE =  'img'
CONFIG_FILE = '../../inst_seg/inst_seg/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml'
MODEL = 'Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'

CCcount = 0

class detic_detector:
    def load_word(self,path='../../inst_seg/inst_seg/datasets/challenge.txt'):
        with open(path, 'r') as f:
            CLASSES = f.readlines()
            CLASSES = list(map(lambda s: s.strip(), CLASSES))
            CLASSES = ','.join(CLASSES)
        self.CLASSES=CLASSES
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
    def __init__(self):
        self.load_word(path='../../inst_seg/inst_seg/datasets/challenge.txt')
        args = AlgorithmConfig()
        args.set(config_file=CONFIG_FILE,
                 confidence_threshold=0.3,
                 vocabulary='custom',  # ['lvis', 'openimages', 'objects365', 'coco', 'custom']
                 custom_vocabulary=self.CLASSES,
                 pred_all_class=False,
                 opts=['MODEL.WEIGHTS', f"../../inst_seg/inst_seg/models/{MODEL}"],
                 input=['../../inst_seg/inst_seg/datasets/03_221214_000014.png'],
                 output='outputs/',
                 input_size=SIZE,
                 acceptable_size_min=0.005,
                 acceptable_size_max=0.2)

        cfg = self.setup_cfg(args)

        self.detector = Predictor(cfg, args)
    def run(self,img):
        predictions, filtered_masks, vis_output = self.detector(img)
        boxes, scores, classes, masks = self.prediction_to_array(predictions)
        return vis_output.get_image(),boxes, scores, classes, masks
    def setup_cfg(self,args):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg, PATH)

        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        # Set score_threshold for builtin models
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH="../../inst_seg/inst_seg/datasets/metadata/lvis_v1_train_cat_info.json"
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand' # load later
        if not args.pred_all_class:
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        cfg.freeze()

        return cfg


