import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertJapaneseTokenizer

TRAIN_JSON_FILENAME = "./Data/train_questions.json"
DEV1_JSON_FILENAME = "./Data/dev1_questions.json"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

TRAIN_FEATURES_DIR = "./Features/Train/"
DEV1_FEATURES_DIR = "./Features/Dev1/"
DEV2_FEATURES_DIR = "./Features/Dev2/"
TRAIN_ALL_FEATURES_DIR = "./AllFeatures/Train/"
DEV1_ALL_FEATURES_DIR = "./AllFeatures/Dev1/"
DEV2_ALL_FEATURES_DIR = "./AllFeatures/Dev2/"

MAX_SEQ_LENGTH = 512

tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


class InputExample(object):
    def __init__(self, qid, question, endings, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, choices_features, label):
        self.choices_features = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


def load_examples(json_filename):
    examples = []

    with open(json_filename, "r", encoding="UTF-8") as r:
        lines = r.read().splitlines()

    for line in lines:
        data = json.loads(line)

        qid = data["qid"]
        question = data["question"].replace("_", "")
        options = data["answer_candidates"][:20]
        answer = data["answer_entity"]

        label = options.index(answer)

        example = InputExample(qid, question, options, label)
        examples.append(example)

    return examples


def convert_example_to_features(example, cache_dir):
    choices_features = []

    # キャッシュファイルからTensorを読み込む。
    for i in range(20):
        directory = cache_dir + example.qid + "/" + str(i) + "/"

        input_ids = torch.load(directory + "input_ids.pt")
        attention_mask = torch.load(directory + "attention_mask.pt")
        token_type_ids = torch.load(directory + "token_type_ids.pt")

        choices_features.append((input_ids, attention_mask, token_type_ids))

    ret = InputFeatures(choices_features, example.label)

    return ret


def select_field(features, field):
    return [
        [choice[field].detach().cpu().numpy() for choice in f.choices_features]
        for f in features
    ]


def create_input_features_dataset(json_filename, cache_dir, save_dir):
    """
    入力特徴量のデータセットを作成します。
    save_dirに空文字列以外を指定すると、処理後のデータをキャッシュファイルに保存します。
    保存されたキャッシュファイルはcreate_input_features_dataset_from_cachesを用いて読み込みます。

    Parameters
    ----------
    json_filename: string
        読み込むJSONファイルのファイル名
    cache_dir: string
        create_features_caches.pyで作成されたキャッシュファイルのディレクトリ名
    save_dir: string
        キャッシュファイルを保存するディレクトリ名
    """

    logger.info("入力特徴量の生成を開始します。")

    examples = load_examples(json_filename)

    features_list = []
    for example in tqdm(examples):
        features = convert_example_to_features(example, cache_dir)
        features_list.append(features)

    logger.info("入力特徴量の生成を完了しました。")

    logger.info("入力する特徴量のデータセットを作成します。")

    all_input_ids = torch.tensor(
        select_field(features_list, "input_ids"), dtype=torch.long
    ).cuda()
    all_input_mask = torch.tensor(
        select_field(features_list, "input_mask"), dtype=torch.long
    ).cuda()
    all_segment_ids = torch.tensor(
        select_field(features_list, "segment_ids"), dtype=torch.long
    ).cuda()
    all_label_ids = torch.tensor(
        [f.label for f in features_list], dtype=torch.long
    ).cuda()

    os.makedirs(save_dir, exist_ok=True)

    torch.save(all_input_ids, save_dir + "all_input_ids.pt")
    torch.save(all_input_mask, save_dir + "all_input_mask.pt")
    torch.save(all_segment_ids, save_dir + "all_segment_ids.pt")
    torch.save(all_label_ids, save_dir + "all_label_ids.pt")

    logger.info("入力する特徴量のデータセットの作成が終了しました。")

if __name__=="__main__":
    create_input_features_dataset(DEV1_JSON_FILENAME,DEV1_FEATURES_DIR,DEV1_ALL_FEATURES_DIR)
