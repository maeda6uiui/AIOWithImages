import os
import sys
import pandas as pd
import torch
from tqdm import tqdm
import cv2
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

#Setup Detectron2
setup_logger()

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)

class ObjectDetection(object):
    def __init__(self,list_filename,image_base_dir):
        #Load the list of articles.
        df = pd.read_table(list_filename, header=None)

        #Make a map of articles.
        self.article_map={}
        for row in df.itertuples(name=None):
            article_name = row[1]
            dir_1 = row[2]
            dir_2 = row[3]

            image_dir = image_base_dir+str(dir_1) + "/" + str(dir_2) + "/"

            self.article_map[article_name]=image_dir

    def get_pred_classes_and_box_centers(self,article_name):
        image_dir=self.article_map[article_name]

        ret_pred_classes=torch.empty(0,dtype=torch.long)
        ret_box_centers=torch.empty(0,dtype=torch.float)

        files = os.listdir(image_dir)
        for file in files:
            im = cv2.imread(image_dir + file)
            outputs = predictor(im)

            pred_classes = outputs["instances"].pred_classes
            pred_classes = pred_classes.flatten()
            ret_pred_classes=torch.cat([ret_pred_classes,pred_classes],dim=0)

            pred_boxes = outputs["instances"].pred_boxes
            box_centers = pred_boxes.get_centers().flatten()
            ret_box_centers=torch.cat([ret_box_centers,box_centers],dim=0)

        return ret_pred_classes,ret_box_centers
