import json
from typing import List, Tuple, Dict

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from video_classification_utils.common import label_to_onehot


class ClipEmbeddingsDataset(Dataset):
    def __init__(
        self, embeddings_dir: str, labels_path: str, average_embeddings: bool = False
    ):
        self.average_embeddings = average_embeddings
        self.embeddings_dir = embeddings_dir

        with open(labels_path, "r") as labels_file:
            self.labels = json.load(labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        video_id, label = self.labels[index]
        with open(f"{self.embeddings_dir}/{video_id}.json", "r") as embedding_file:
            embeddings_list = json.load(embedding_file)

        embeddings = torch.tensor(embeddings_list)
        if self.average_embeddings:
            embeddings = torch.mean(embeddings, dim=0)

        return embeddings, label_to_onehot(label)

    def collate_function(
        self, batch: List[Tuple[torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        embeddings = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        if self.average_embeddings:
            return {
                "embeddings": torch.stack(embeddings),
                "labels": torch.stack(labels),
            }
        else:
            batched_embeddings = pad_sequence(embeddings, batch_first=True).float()
            batch_size, seq_len, _ = batched_embeddings.shape
            attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.float)
            for i, seq in enumerate(embeddings):
                attention_mask[i, : len(embeddings)] = 1

            return {
                "embeddings": batched_embeddings,
                "attention_mask": attention_mask,
                "labels": torch.stack(labels),
            }
