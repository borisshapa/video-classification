from typing import Dict

import torch

CATEGORY_TO_ID = {"food": 0, "art_music": 1, "travel": 2, "history": 3}


def dict_to_device(
    dict: Dict[str, torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    for key, value in dict.items():
        dict[key] = value.to(device)
    return dict


def label_to_onehot(label: int) -> torch.Tensor:
    one_hot = torch.zeros(len(CATEGORY_TO_ID))
    one_hot[CATEGORY_TO_ID[label]] = 1
    return one_hot
