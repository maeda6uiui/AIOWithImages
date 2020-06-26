import os
import torch

TRAIN_ALL_FEATURES_DIR = "./AllFeatures/Train/"
DEV1_ALL_FEATURES_DIR = "./AllFeatures/Dev1/"
DEV2_ALL_FEATURES_DIR = "./AllFeatures/Dev2/"

TRAIN_RESHAPED_FEATURES_DIR = "./ReshapedFeatures/Train/"
DEV1_RESHAPED_FEATURES_DIR = "./ReshapedFeatures/Dev1/"
DEV2_RESHAPED_FEATURES_DIR = "./ReshapedFeatures/Dev2/"

IMAGE_FEATURES_LENGTH = 200


def reshape(cache_dir, save_dir, end_index):
    all_input_ids = torch.load(cache_dir + "all_input_ids.pt")
    all_input_mask = torch.load(cache_dir + "all_input_mask.pt")
    all_segment_ids = torch.load(cache_dir + "all_segment_ids.pt")
    all_label_ids = torch.load(cache_dir + "all_label_ids.pt")

    all_input_ids = all_input_ids[:, :, :end_index]
    all_input_mask = all_input_mask[:, :, :end_index]
    all_segment_ids = all_segment_ids[:, :, :end_index]

    os.makedirs(save_dir, exist_ok=True)

    torch.save(all_input_ids, save_dir + "all_input_ids.pt")
    torch.save(all_input_mask, save_dir + "all_input_mask.pt")
    torch.save(all_segment_ids, save_dir + "all_segment_ids.pt")
    torch.save(all_label_ids, save_dir + "all_label_ids.pt")


if __name__ == "__main__":
    reshape(TRAIN_ALL_FEATURES_DIR, TRAIN_RESHAPED_FEATURES_DIR, IMAGE_FEATURES_LENGTH)
    reshape(DEV1_ALL_FEATURES_DIR, DEV1_RESHAPED_FEATURES_DIR, IMAGE_FEATURES_LENGTH)
    reshape(DEV2_ALL_FEATURES_DIR, DEV2_RESHAPED_FEATURES_DIR, IMAGE_FEATURES_LENGTH)
