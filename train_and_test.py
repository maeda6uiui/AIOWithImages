import json
import logging
import os
import sys
from tqdm import tqdm
import numpy as np
import torch
from transformers import (
    BertJapaneseTokenizer,
    BertForMultipleChoice,
    AdamW,
    get_linear_schedule_with_warmup,
)

TRAIN_JSON_FILENAME = "./Data/train_questions.json"
DEV1_JSON_FILENAME = "./Data/dev1_questions.json"
DEV2_JSON_FILENAME = "./Data/dev2_questions.json"

TRAIN_FEATURES_DIR = "./Features/Train/"
DEV1_FEATURES_DIR = "./Features/Dev1/"
DEV2_FEATURES_DIR = "./Features/Dev2/"
TRAIN_ALL_FEATURES_DIR = "./AllFeatures/Train/"
DEV1_ALL_FEATURES_DIR = "./AllFeatures/Dev1/"
DEV2_ALL_FEATURES_DIR = "./AllFeatures/Dev2/"

EPOCH_NUM = 3
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE = 4

MAX_SEQ_LENGTH = 512
INPUT_SEQ_LENGTH = 200

tokenizer = BertJapaneseTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-whole-word-masking"
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def create_input_features_dataset_from_caches(cache_dir):
    """
    キャッシュファイルを読み込み、入力特徴量のデータセットを作成します。

    Parameters
    ----------
    cache_dir: string
        キャッシュファイルのディレクトリ名

    Returns
    ----------
    dataset: TensorDataset
        BERTモデルに入力する特徴量のデータセット
    """

    logger.info("入力する特徴量のデータセットを作成します。")

    all_input_ids = torch.load(cache_dir + "all_input_ids.pt")
    all_input_mask = torch.load(cache_dir + "all_input_mask.pt")
    all_segment_ids = torch.load(cache_dir + "all_segment_ids.pt")
    all_label_ids = torch.load(cache_dir + "all_label_ids.pt")

    # tensorがメモリに乗らないので、サイズを小さくする。
    all_input_ids = all_input_ids[:, :, :INPUT_SEQ_LENGTH]
    all_input_mask = all_input_mask[:, :, :INPUT_SEQ_LENGTH]
    all_segment_ids = all_segment_ids[:, :, :INPUT_SEQ_LENGTH]

    # clamp
    all_input_ids = torch.clamp(all_input_ids, 0, len(tokenizer) - 1)

    dataset = torch.utils.data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    logger.info("入力する特徴量のデータセットの作成が終了しました。")

    return dataset


def train(model, train_dataset):
    logger.info("訓練を開始します。")
    logger.info("バッチサイズ: {}".format(TRAIN_BATCH_SIZE))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True
    )

    model.train()

    lr = 5e-5
    eps = 1e-8
    logger.info("lr = {}".format(lr))
    logger.info("eps = {}".format(eps))

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    total_steps = len(train_dataloader) * EPOCH_NUM
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    log_interval = 100

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # パラメータの更新
            optimizer.step()
            scheduler.step()

            model.zero_grad()

            if step % log_interval == 0:
                logger.info("損失: {}".format(loss.item()))

    torch.save(model.state_dict(), "./pytorch_model.bin")

    logger.info("訓練を終了しました。")


def test(model, test_dataset):
    logger.info("テストを開始します。")
    logger.info("バッチサイズ: {}".format(TEST_BATCH_SIZE))

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


def output_memory_allocation_info():
    logger.debug(
        "Current memory allocated: {} MB".format(
            torch.cuda.memory_allocated() / 1024 ** 2
        )
    )
    logger.debug("Cached memory: {} MB".format(torch.cuda.memory_cached() / 1024 ** 2))


if __name__ == "__main__":
    # モデルの作成
    model = BertForMultipleChoice.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )
    model.cuda()

    # finetuningされたパラメータを読み込む。
    # model.load_state_dict(torch.load("./pytorch_model.bin"))

    train_dataset = create_input_features_dataset_from_caches(TRAIN_ALL_FEATURES_DIR)
    train(model, train_dataset)

    test_dataset = create_input_features_dataset_from_caches(DEV2_ALL_FEATURES_DIR)
    test(model, test_dataset)
