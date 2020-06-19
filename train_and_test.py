import json
import logging
from tqdm import tqdm

import numpy as np
import torch

from transformers import BertJapaneseTokenizer, BertForMultipleChoice

TRAIN_JSON_FILENAME = "./Data/train_questions.json"
DEV1_JSON_FILENAME = "./Data/dev1_questions.json"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

TRAIN_FEATURES_DIR = "./Features/Train/"
DEV1_FEATURES_DIR = "./Features/Dev1/"
DEV2_FEATURES_DIR = "./Features/Dev2/"

EPOCH_NUM = 5
BATCH_SIZE = 128

MAX_SEQ_LENGTH = 512

tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

logger = logging.getLogger("awi")


class InputExample(object):
    def __init__(self, example_id, question, endings, label=None):
        self.example_id = example_id
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


class InputFeaturesDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.input_features_list = []

    def append(self, input_features):
        self.input_features_list.append(input_features)

    def __len__(self):
        return len(self.input_features_list)

    def __getitem__(self, index):
        return self.input_features_list[index]


def load_examples(json_filename, cache_dir):
    examples = []

    with open(json_filename, "r", encoding="UTF-8") as r:
        lines = r.read().splitlines()

    for line in tqdm(lines):
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
    # 画像の特徴量を読み込む。
    image_features_filename = cache_dir + example.qid + "/" + "image_features.pt"
    image_features = torch.load(image_features_filename)
    image_features_length = image_features.size()[0]

    choices_features = []

    # 選択肢それぞれについて処理を行う。
    for ending in example.endings:
        # tokenizerの入力は「問題文」+「選択肢」
        input_text = example.question + "[SEP]" + ending

        # tokenizerを用いてencoding
        encoding = tokenizer.encode_plus(
            input_text,
            return_tensors="pt",
            add_special_tokens=True,
            pad_to_max_length=False,
        )

        input_ids = encoding["input_ids"].cuda().float()
        text_features_length = input_ids.size()[0]  # input_idsのうちテキスト部分の長さ

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
            zero_padding = torch.zeros(padding_length).cuda().float()
            input_ids = torch.cat([input_ids, zero_padding], dim=0)

            for i in range(
                text_features_length + image_features_length, MAX_SEQ_LENGTH
            ):
                attention_mask[i] = 0

        # token_type_idsの作成
        # テキスト: 0 画像: 1
        token_type_ids = torch.zeros(MAX_SEQ_LENGTH).cuda()
        for i in range(0, text_features_length):
            token_type_ids[i] = 0
        for i in range(text_features_length, MAX_SEQ_LENGTH):
            token_type_ids[i] = 1

        choices_features.append((input_ids, attention_mask, token_type_ids))

    ret = InputFeatures(choices_features, example.label)

    return ret


def create_input_features_dataset(json_filename, cache_dir):
    logger.info("入力特徴量の生成を開始します。")

    examples = load_examples(json_filename, cache_dir)

    ret = InputFeaturesDataset()
    for example in examples:
        features = convert_example_to_features(example, cache_dir)
        ret.append(features)

    logging.info("入力特徴量の生成を完了しました。")

    return ret


def train(model, train_dataset):
    logger.info("訓練を開始します。")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    log_interval = 5

    model.train()

    for epoch in range(EPOCH_NUM):
        logger.info("========== Epoch {} / {} ==========".format(epoch + 1, EPOCH_NUM))

        for step, batch in enumerate(tqdm(train_dataloader)):
            batch = tuple(t for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }

            # 勾配の初期化
            optimizer.zero_grad()
            # 順伝播
            outputs = model(**inputs)
            loss = outputs[0]
            # 逆伝播
            loss.backward()
            # パラメータの更新
            optimizer.step()

            model.zero_grad()

            if step % log_interval == 0:
                logger.info(
                    "進捗: {:.2f} %\t損失: {}".format(
                        step * len(batch) / len(train_dataloader), loss.item()
                    )
                )

    torch.save(model.state_dict(), "./pytorch_model.bin")

    logger.info("訓練を終了しました。")


def test(model, test_dataset):
    logger.info("テストを開始します。")

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    model.eval()

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for step, batch in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            batch = tuple(t for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3],
            }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    pred_ids = np.argmax(preds, axis=1)

    accuracy = simple_accuracy(pred_ids, out_label_ids)

    logger.info("テストが終了しました。")
    logger.info("損失: {}\n精度: {}".format(eval_loss, accuracy))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


if __name__ == "__main__":
    # モデルの作成
    model = BertForMultipleChoice.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    model.cuda()

    # finetuningされたパラメータを読み込む。
    # model.load_state_dict(torch.load("./pytorch_model.bin"))

    train_dataset = create_input_features_dataset(
        TRAIN_JSON_FILENAME, TRAIN_FEATURES_DIR
    )
    train(model, train_dataset)

    test_dataset = create_input_features_dataset(DEV2_JSON_FILENAME, DEV2_FEATURES_DIR)
    test(model, test_dataset)
