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

IMAGE_DIR = "./WikipediaImages/Images/"
FEATURES_DIR = "./WikipediaImages/Features/"

# article_listの読み込み
df = pd.read_table("./WikipediaImages/article_list.txt", header=None)

# Detectron2のセットアップ
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
predictor = DefaultPredictor(cfg)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class ImageInfo(object):
    def __init__(self, image_dir, article_name):
        self.image_dir = image_dir
        self.article_name = article_name


def get_image_info_list():
    ret = []

    for row in df.itertuples(name=None):
        article_name = row[1]
        dir_1 = row[2]
        dir_2 = row[3]

        image_dir = IMAGE_DIR + str(dir_1) + "/" + str(dir_2) + "/"

        image_info = ImageInfo(image_dir, article_name)
        ret.append(image_info)

    return ret


def get_image_features(image_dir):
    ret = torch.empty(0).cuda()

    files = os.listdir(image_dir)
    for file in files:
        im = cv2.imread(image_dir + file)
        outputs = predictor(im)

        pred_classes = outputs["instances"].pred_classes
        pred_classes = pred_classes.flatten()

        pred_boxes = outputs["instances"].pred_boxes
        box_areas = pred_boxes.area().long().flatten()

        tmp = torch.cat([pred_classes, box_areas], dim=0)
        ret = torch.cat([ret, tmp], dim=0)

    return ret


def create_image_features():
    logger.info("画像特徴量の生成を開始しました。")

    image_info_list = get_image_info_list()

    os.makedirs(FEATURES_DIR, exist_ok=True)

    for i, image_info in enumerate(tqdm(image_info_list)):
        image_features = torch.zeros(0).cuda()
        if image_info.image_dir != "":
            try:
                image_features = get_image_features(image_info.image_dir)
            except AssertionError as e:
                logger.error(e)

        directory = FEATURES_DIR + "/" + image_info.article_name + "/"
        os.makedirs(directory, exist_ok=True)

        torch.save(image_features, directory + "image_features.pt")

    logger.info("画像特徴量の生成を終了しました。")


if __name__ == "__main__":
    create_image_features()
