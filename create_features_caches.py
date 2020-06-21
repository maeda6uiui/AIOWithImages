import json
import logging
import os
from tqdm import tqdm
import torch
from transformers import BertJapaneseTokenizer

TRAIN_JSON_FILENAME = "./Data/train_questions.json"
DEV1_JSON_FILENAME = "./Data/dev1_questions.json"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

IMAGE_FEATURES_DIR = "./WikipediaImages/Features/"
TRAIN_FEATURES_DIR = "./Features/Train/"
DEV1_FEATURES_DIR = "./Features/Dev1/"
DEV2_FEATURES_DIR = "./Features/Dev2/"

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
    """
    exampleを特徴量に変換してキャッシュファイルに保存します。
    """

    # 選択肢それぞれについて処理を行う。
    for i, ending in enumerate(example.endings):
        # tokenizerの入力は「問題文」+「選択肢」
        input_text = example.question + "[SEP]" + ending

        # tokenizerを用いてencoding
        encoding = tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            add_special_tokens=True,
            pad_to_max_length=False,
        )

        input_ids = encoding["input_ids"].cuda()
        input_ids = input_ids.view(-1)
        text_features_length = input_ids.size()[0]  # input_idsのうちテキスト部分の長さ

        # 画像の特徴量を読み込む。
        article_name = example.endings[example.label]
        image_features_filename = (
            IMAGE_FEATURES_DIR + article_name + "/" + "image_features.pt"
        )

        image_features = None
        if os.path.exists(image_features_filename) == True:
            image_features = torch.load(image_features_filename)
        else:
            image_features = torch.zeros(0, dtype=torch.long).cuda()

        image_features_length = image_features.size()[0]

        # 画像の特徴量を結合する。
        input_ids = torch.cat([input_ids, image_features], dim=0)
        attention_mask = torch.ones(MAX_SEQ_LENGTH).cuda()
        # BERTモデルに入力する特徴量が長すぎる場合には、オーバーする部分を切り捨てる。
        if text_features_length + image_features_length > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
        # それ以外の場合には、足りない部分を0で埋める。
        else:
            padding_length = MAX_SEQ_LENGTH - (
                text_features_length + image_features_length
            )
            zero_padding = torch.zeros(padding_length, dtype=torch.long).cuda()
            input_ids = torch.cat([input_ids, zero_padding], dim=0)

            for j in range(
                text_features_length + image_features_length, MAX_SEQ_LENGTH
            ):
                attention_mask[j] = 0

        # token_type_idsの作成
        # テキスト: 0 画像: 1
        token_type_ids = torch.zeros(MAX_SEQ_LENGTH, dtype=torch.long).cuda()
        for j in range(0, text_features_length):
            token_type_ids[j] = 0
        for j in range(text_features_length, MAX_SEQ_LENGTH):
            token_type_ids[j] = 1

        # Tensorを保存する。
        directory = cache_dir + example.qid + "/" + str(i) + "/"
        os.makedirs(directory, exist_ok=True)

        torch.save(input_ids, directory + "input_ids.pt")
        torch.save(attention_mask, directory + "attention_mask.pt")
        torch.save(token_type_ids, directory + "token_type_ids.pt")


if __name__ == "__main__":
    # 訓練データ
    logger.info("訓練データの特徴量の生成を開始します。")

    examples=load_examples(TRAIN_JSON_FILENAME)
    for example in tqdm(examples):
        convert_example_to_features(example,TRAIN_FEATURES_DIR)

    logger.info("訓練データの特徴量の生成が終了しました。")

    # テストデータ
    logger.info("テストデータの特徴量の生成を開始します。")

    examples = load_examples(DEV2_JSON_FILENAME)
    for example in tqdm(examples):
        convert_example_to_features(example, DEV2_FEATURES_DIR)

    logger.info("テストデータの特徴量の生成が終了しました。")
