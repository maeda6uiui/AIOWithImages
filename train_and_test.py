import json
import logging
import os
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForMultipleChoice

TRAIN_JSON_FILENAME = "./Data/train_questions.json"
DEV1_JSON_FILENAME = "./Data/dev1_questions.json"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

IMAGE_FEATURES_DIR = "./WikipediaImages/Features/"
TRAIN_FEATURES_DIR = "./Features/Train/"
DEV1_FEATURES_DIR = "./Features/Dev1/"
DEV2_FEATURES_DIR = "./Features/Dev2/"

EPOCH_NUM = 5
TRAIN_BATCH_SIZE = 1
TEST_BATCH_SIZE = 4

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


def convert_example_to_features(example,cache_dir):
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


def create_input_features_dataset(json_filename, cache_dir):
    """
    入力特徴量のデータセットを作成します。

    Parameters
    ----------
    json_filename: string
        読み込むJSONファイルのファイル名
    cache_dir: string
        キャッシュファイルのディレクトリ名

    Returns
    ----------
    dataset: TensorDataset
        BERTモデルに入力する特徴量のデータセット
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

    dataset = torch.utils.data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    logger.info("入力する特徴量のデータセットの作成が終了しました。")

    return dataset


def train(model, train_dataset):
    logger.info("訓練を開始します。")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
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
        test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False
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
    logger.info("損失: {}\n正確度: {}".format(eval_loss, accuracy))


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

    test_dataset = create_input_features_dataset(
        DEV2_JSON_FILENAME, DEV2_FEATURES_DIR
    )
    test(model, test_dataset)
