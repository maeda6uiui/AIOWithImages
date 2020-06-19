import json
import logging
import os
import pandas as pd
from tqdm import tqdm

import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

TRAIN_JSON_FILENAME="./Data/train_questions.json"
DEV1_JSON_FILENAME="./Data/dev1_questions.json"
DEV2_JSON_FILENAME="./Data/dev2_questions.json"

IMAGE_DIR="./WikipediaImages/Images/"

TRAIN_FEATURES_DIR="./Features/Train/"
DEV1_FEATURES_DIR="./Features/Dev1/"
DEV2_FEATURES_DIR="./Features/Dev2/"

#article_listの読み込み
df=pd.read_table("./WikipediaImages/article_list.txt",header=None)

#Detectron2のセットアップ
cfg=get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
cfg.MODEL.WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor=DefaultPredictor(cfg)

logger=logging.getLogger("awi")

class ImageInfo(object):
    def __init__(self,image_dir,qid):
        self.image_dir=image_dir
        self.qid=qid

def get_image_info_list(json_filename):
    logger.info("画像ディレクトリ一覧の取得を開始しました。")

    ret=[]

    with open(json_filename,"r",encoding="UTF-8") as r:
        lines=r.read().splitlines()

    for line in tqdm(lines):
        data=json.loads(line)

        qid=data["qid"]
        answer_entity=data["answer_entity"]

        image_location_info=df[df[0]==answer_entity]
        if len(image_location_info)==0:
            continue

        image_dir=IMAGE_DIR+str(image_location_info.iat[0,1])+"/"+str(image_location_info.iat[0,2])+"/"
        
        image_info=ImageInfo(image_dir,qid)
        ret.append(image_info)

    logger.info("画像ディレクトリ一覧の取得を終了しました。")

    return ret

def create_image_features(image_info_list,cache_dir):
    os.makedirs(cache_dir,exist_ok=True)

    for i,image_info in enumerate(tqdm(image_info_list)):
        image_features=torch.zeros(0).cuda().float()
        if image_info.image_dir!="":
            try:
                image_features=get_image_features(image_info.image_dir)
            except AssertionError as e:
                logger.error(e)

        directory=cache_dir+"/"+image_info.qid+"/"
        os.makedirs(directory,exist_ok=True)

        torch.save(image_features,directory+"image_features.pt")

def get_image_features(image_dir):
    ret=torch.empty(0).cuda().float()

    files=os.listdir(image_dir)
    for file in files:
        im=cv2.imread(image_dir+file)
        outputs=predictor(im)

        pred_classes=outputs["instances"].pred_classes
        pred_classes=pred_classes.flatten().float()

        pred_boxes=outputs["instances"].pred_boxes
        box_areas=pred_boxes.area().flatten()

        tmp=torch.cat([pred_classes,box_areas],dim=0)
        ret=torch.cat([ret,tmp],dim=0)

    return ret

if __name__=="__main__":
    train_image_dirs=get_image_info_list(TRAIN_JSON_FILENAME)
    dev1_image_dirs=get_image_info_list(DEV1_JSON_FILENAME)
    dev2_image_dirs=get_image_info_list(DEV2_JSON_FILENAME)

    create_image_features(train_image_dirs,TRAIN_FEATURES_DIR)
    create_image_features(dev1_image_dirs,DEV1_FEATURES_DIR)
    create_image_features(dev2_image_dirs,DEV2_FEATURES_DIR)
