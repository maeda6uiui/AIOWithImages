import logging
import os
import pandas as pd
import torch
import torchvision
from tqdm import tqdm
from PIL import Image

vgg16 = torchvision.models.vgg16(pretrained=True)
vgg16.eval()
vgg16.classifier[6] = torch.nn.Linear(4096, 50)

normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize,
    ]
)

IMAGE_DIR = "./WikipediaImages/Images/"
FEATURES_DIR = "./WikipediaImages/Features/"

# article_listの読み込み
df = pd.read_table("./WikipediaImages/article_list.txt", header=None)

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
    ret = torch.empty(0, dtype=torch.long).cuda()

    files = os.listdir(image_dir)
    for file in files:
        try:
            img = Image.open(image_dir + file)
        except:
            logger.error("Failed to open {}.".format(image_dir+file))
            continue

        img = img.convert("RGB")
        img_tensor = preprocess(img)
        img_tensor = img_tensor.unsqueeze(0)

        features_tensor = vgg16(img_tensor)

        SCALE = 10000
        features_tensor = (features_tensor - torch.min(features_tensor)) * SCALE
        features_tensor = features_tensor.long().flatten().cuda()

        ret = torch.cat([ret, features_tensor], dim=0)

    return ret


def create_image_features():
    logger.info("画像特徴量の生成を開始しました。")

    image_info_list = get_image_info_list()

    os.makedirs(FEATURES_DIR, exist_ok=True)

    for i, image_info in enumerate(tqdm(image_info_list)):
        image_features = torch.zeros(0, dtype=torch.long).cuda()
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
