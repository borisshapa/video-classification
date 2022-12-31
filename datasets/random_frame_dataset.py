import hashlib
import json
from typing import List, Tuple, Callable

import cv2
import torch
from PIL import Image
from torch.utils.data import Dataset

from video_classification_utils.common import label_to_onehot


class RandomFrameDataset(Dataset):
    def __init__(
        self,
        labels_path: str,
        video_dir: str,
        preprocess: Callable[[Image.Image], torch.Tensor],
    ):
        with open(labels_path, "r") as labels_file:
            self.labels = json.load(labels_file)
        self.video_dir = video_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        video_id, label = self.labels[index]
        video_path = f"{self.video_dir}/{video_id}.mp4"
        video_capture = cv2.VideoCapture(video_path)
        frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or video_capture.get(
            cv2.cv.CV_CAP_PROP_FRAME_COUNT
        )

        video_id_hash = int(hashlib.sha256(video_id.encode("utf-8")).hexdigest(), 16)
        frame_id = video_id_hash % frame_count
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        success, frame = video_capture.read()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = self.preprocess(Image.fromarray(rgb_frame).convert("RGB"))
        return tensor_frame, label_to_onehot(label)

    def collate_function(
        self, batch: List[Tuple[torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        return torch.stack(images), torch.stack(labels)
