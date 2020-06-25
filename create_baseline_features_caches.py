import gzip
import json
import logging
import os
from tqdm import tqdm
import torch
from transformers import BertJapaneseTokenizer

CANDIDATE_ENTITIES_FILENAME = "./Data/candidate_entities.json.gz"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

DEV2_FEATURES_DIR = "./BaselineFeatures/Dev2/"

MAX_SEQ_LENGTH = 512

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)


class InputExample(object):
    def __init__(self, qid, question, endings, contexts, label=None):
        self.qid = qid
        self.question = question
        self.endings = endings
        self.contexts = contexts
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


def load_contexts(gz_filename):
    ret = {}

    with gzip.open(gz_filename, mode="rt", encoding="UTF-8") as f:
        for line in f:
            data = json.loads(line)

            title = data["title"]
            text = data["text"]

            ret[title] = text

    return ret


def load_examples(json_filename):
    logger.info("contextsの読み込みを開始します。")
    candidate_entities = load_contexts(CANDIDATE_ENTITIES_FILENAME)
    logger.info("contextsの読み込みが終了しました。")

    logger.info("examplesの読み込みを開始します。")

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

        contexts = []
        for option in options:
            context = candidate_entities[option]
            contexts.append(context)

        example = InputExample(qid, question, options, contexts, label)
        examples.append(example)

    logger.info("examplesの読み込みが終了しました。。")

    return examples


def convert_example_to_features(example):
    choices_features = []

    for i in range(20):
        text_a = example.contexts[i]
        text_b = example.question + "[SEP]" + example.endings[i]

        inputs = tokenizer.encode_plus(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=MAX_SEQ_LENGTH,
            truncation_strategy="only_first",  # 常にcontextをtruncate
        )

        input_ids, token_type_ids = (
            inputs["input_ids"],
            inputs["token_type_ids"],
        )

        attention_mask = [0] * len(input_ids)

        padding_length = MAX_SEQ_LENGTH - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(attention_mask) == MAX_SEQ_LENGTH
        assert len(token_type_ids) == MAX_SEQ_LENGTH

        input_ids = torch.tensor(input_ids, dtype=torch.long).cuda()
        attention_mask = torch.tensor(attention_mask, dtype=torch.long).cuda()
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).cuda()

        choices_features.append((input_ids, attention_mask, token_type_ids))

    ret = InputFeatures(choices_features, example.label)

    return ret


def select_field(features, field):
    return [
        [choice[field].detach().cpu().numpy() for choice in f.choices_features]
        for f in features
    ]


def create_input_features_dataset(json_filename, save_dir=""):
    logger.info("入力特徴量の生成を開始します。")

    examples = load_examples(json_filename)

    features_list = []
    for example in tqdm(examples):
        features = convert_example_to_features(example)
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

    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

        torch.save(all_input_ids, save_dir + "all_input_ids.pt")
        torch.save(all_input_mask, save_dir + "all_input_mask.pt")
        torch.save(all_segment_ids, save_dir + "all_segment_ids.pt")
        torch.save(all_label_ids, save_dir + "all_label_ids.pt")

    dataset = torch.utils.data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    logger.info("入力する特徴量のデータセットの作成が終了しました。")

    return dataset


if __name__ == "__main__":
    create_input_features_dataset(DEV2_JSON_FILENAME, DEV2_FEATURES_DIR)
