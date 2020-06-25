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
)

DEV2_BASELINE_FEATURES_DIR="./BaselineFeatures/Dev2/"
DEV2_ALL_FEATURES_DIR = "./AllFeatures/Dev2/"

BASELINE_MODEL_FILENAME="./Model/Baseline/pytorch_model.bin"
IMAGE_MODEL_FILENAME="./Model/VGG16/pytorch_model.bin"

TEST_BATCH_SIZE = 4

MAX_SEQ_LENGTH = 512
INPUT_SEQ_LENGTH = 200

SCORE_THRESHOLD=4.0

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

model=None
image_model=None
tokenizer=None

def init():
    global model
    global image_model
    global tokenizer

    model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    model.load_state_dict(torch.load(BASELINE_MODEL_FILENAME))
    model.cuda()

    image_model=BertForMultipleChoice.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
    image_model.load_state_dict(torch.load(IMAGE_MODEL_FILENAME))
    image_model.cuda()

    tokenizer=BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

def load_baseline_dataset(cache_dir):
    logger.info("入力する特徴量のデータセットを作成します。")

    all_input_ids = torch.load(cache_dir + "all_input_ids.pt")
    all_input_mask = torch.load(cache_dir + "all_input_mask.pt")
    all_segment_ids = torch.load(cache_dir + "all_segment_ids.pt")
    all_label_ids = torch.load(cache_dir + "all_label_ids.pt")

    dataset = torch.utils.data.TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )

    logger.info("入力する特徴量のデータセットの作成が終了しました。")

    return dataset

def load_image_dataset(cache_dir):
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

def test(test_dataset,image_dataset):
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
    
    """
    #画像データを使ってテスト
    image_dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=1, shuffle=False
    )

    pred_ids=[]

    for step,batch in enumerate(tqdm(image_dataloader)):
        preds_row=preds[step]
        scores=np.sort(preds_row)[::-1]
        if scores[0]-scores[1]>SCORE_THRESHOLD:
            pred_ids.append(np.argmax(preds_row))
            continue

        batch = tuple(t for t in batch)
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "labels": batch[3],
        }

        with torch.no_grad():
            outputs=image_model(**inputs)
        
        logits=outputs[1]
        logits=logits.detach().cpu().numpy()
        pred_index=np.argmax(logits)

        pred_ids.append(pred_index)
    """

    pred_ids = np.argmax(preds, axis=1)

    #pred_ids=np.array(pred_ids)

    accuracy = simple_accuracy(pred_ids, out_label_ids)

    logger.info("テストが終了しました。")
    logger.info("正確度: {}".format(accuracy))


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

if __name__=="__main__":
    init()

    baseline_dataset=load_baseline_dataset(DEV2_BASELINE_FEATURES_DIR)
    image_dataset=load_image_dataset(DEV2_ALL_FEATURES_DIR)
    test(baseline_dataset,image_dataset)
