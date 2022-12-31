from typing import List, Callable

import clip
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.functional as F

from datasets.clip_embeddings_dataset import ClipEmbeddingsDataset
from datasets.random_frame_dataset import RandomFrameDataset
from video_classification_utils.common import dict_to_device
from IPython import display


def cross_entropy(input, target):
    log_softmax = F.log_softmax(input)
    batch_size = input.shape[0]
    return -(target * log_softmax).sum() / batch_size


def draw_plots(loss_history: List[float], accuracy: List[float], f1: List[float]):
    display.clear_output(wait=True)

    f, (ax1, ax2, ax3) = plt.subplots(3)
    f.set_figwidth(15)
    f.set_figheight(10)

    ax1.set_title("training loss")
    ax2.set_title("accuracy")
    ax3.set_title("f1")

    ax1.plot(loss_history)
    ax2.plot(accuracy)
    ax3.plot(f1)

    plt.show()

    if len(loss_history) > 0:
        print(f"Current loss: {loss_history[-1]}")
    if len(accuracy) > 0:
        print(f"Current accuracy: {accuracy[-1]}")
        print(f"Current f1: {f1[-1]}")


def train_on_embeddings_dataset(
    model: nn.Module,
    average_embeddings: bool,
    mixup: bool,
    batch_size: int,
    epochs: int,
    alpha: float,
    device: str,
    log_every: int,
):
    train_dataset = ClipEmbeddingsDataset(
        embeddings_dir="resources/data/embeddings",
        labels_path="resources/data/train_config.json",
        average_embeddings=average_embeddings,
    )
    test_dataset = ClipEmbeddingsDataset(
        embeddings_dir="resources/data/embeddings",
        labels_path="resources/data/test_config.json",
        average_embeddings=average_embeddings,
    )
    train_data_loader1 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )
    train_data_loader2 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_function,
    )

    optimizer = Adam(model.parameters())

    step = 0

    loss_history, accuracy, f1 = [], [], []

    for epoch in range(1, epochs + 1):
        iterator = (
            zip(train_data_loader1, train_data_loader2) if mixup else train_data_loader1
        )
        for batch in iterator:
            step += 1

            optimizer.zero_grad()

            if mixup:
                lam = np.random.beta(alpha, alpha)
                batch1, batch2 = batch
                batch = {
                    "embeddings": lam * batch1["embeddings"]
                    + (1 - lam) * batch2["embeddings"],
                    "labels": lam * batch1["labels"] + (1 - lam) * batch2["labels"],
                }
                if "attention_mask" in batch1:
                    batch["attention_mask"] = torch.logical_or(
                        batch1["attention_mask"].bool(), batch2["attention_mask"].bool()
                    ).float()

            dict_to_device(batch, device)

            labels = batch["labels"]
            del batch["labels"]
            predictions = model(**batch)
            loss = cross_entropy(predictions, labels)
            loss.backward()

            loss_history.append(loss.item())
            optimizer.step()

            if step % log_every == 0:
                model.eval()
                with torch.no_grad():
                    predictions = []
                    ground_truth = []

                    for batch in test_data_loader:
                        dict_to_device(batch, device)
                        labels = batch["labels"].argmax(dim=1)
                        del batch["labels"]
                        prediction = model(**batch).argmax(dim=1)

                        ground_truth.append(labels.cpu().detach())
                        predictions.append(prediction.cpu().detach())

                    ground_truth = torch.cat(ground_truth).numpy()
                    predictions = torch.cat(predictions).numpy()

                    accuracy.append(accuracy_score(ground_truth, predictions))
                    f1.append(f1_score(ground_truth, predictions, average="micro"))
                model.train()

            draw_plots(loss_history, accuracy, f1)
            print(f"Epoch {epoch} / {epochs}")


def train_on_images_dataset(
    model: nn.Module,
    mixup: bool,
    batch_size: int,
    epochs: int,
    alpha: float,
    device: str,
    log_every: int,
):
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    train_dataset = RandomFrameDataset(
        labels_path="resources/data/train_config.json",
        video_dir="resources/data/videos",
        preprocess=preprocess,
    )
    test_dataset = RandomFrameDataset(
        labels_path="resources/data/test_config.json",
        video_dir="resources/data/videos",
        preprocess=preprocess,
    )
    train_data_loader1 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )
    train_data_loader2 = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_function,
    )
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_function,
    )

    optimizer = Adam(model.parameters())

    step = 0

    loss_history, accuracy, f1 = [], [], []

    for epoch in range(1, epochs + 1):
        iterator = (
            zip(train_data_loader1, train_data_loader2) if mixup else train_data_loader1
        )
        for batch in iterator:
            step += 1

            optimizer.zero_grad()

            if mixup:
                lam = np.random.beta(alpha, alpha)
                (images1, labels1), (images2, labels2) = batch
                images = lam * images1 + (1 - lam) * images2
                labels = lam * labels1 + (1 - lam) * labels2
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)
            embeddings = clip_model.encode_image(images).float()
            predictions = model(embeddings)
            loss = cross_entropy(predictions, labels)
            loss.backward()

            loss_history.append(loss.item())
            optimizer.step()

            if step % log_every == 0:
                model.eval()
                with torch.no_grad():
                    predictions = []
                    ground_truth = []

                    for images, labels in test_data_loader:
                        labels = labels.argmax(dim=1).to(device)
                        images = images.to(device)
                        embeddings = clip_model.encode_image(images).float()
                        prediction = model(embeddings).argmax(dim=1)

                        ground_truth.append(labels.cpu().detach())
                        predictions.append(prediction.cpu().detach())

                    ground_truth = torch.cat(ground_truth).numpy()
                    predictions = torch.cat(predictions).numpy()

                    accuracy.append(accuracy_score(ground_truth, predictions))
                    f1.append(f1_score(ground_truth, predictions, average="micro"))
                model.train()

            draw_plots(loss_history, accuracy, f1)
            print(f"Epoch {epoch} / {epochs}")
