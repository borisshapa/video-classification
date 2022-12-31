import csv
import json
import os
from argparse import ArgumentParser, Namespace
from collections import Counter

import clip
import torch
from loguru import logger

from pytube import YouTube
from tqdm import tqdm

from video_classification_utils.video import get_frames_from_video


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--youtube-csv-file",
        type=str,
        default="resources/data/youtube.csv",
        help="path to youtube videos dataset (can be downloaded here: https://www.kaggle.com/datasets/rajatrc1705/youtube-videos-dataset)",
    )

    arg_parser.add_argument(
        "--save-videos-to",
        type=str,
        default="resources/data/videos",
        help="path to the directory where the youtube videos will be saved",
    )
    arg_parser.add_argument(
        "--save-config-to",
        type=str,
        default="resources/data/config.json",
        help="path to the file where the config with the mapping from video id to category will be saved",
    )
    arg_parser.add_argument(
        "--frames-count",
        type=int,
        default=100,
        help="number of frames based on which to receive video embedding",
    )
    arg_parser.add_argument(
        "--save-embeddings-to",
        type=str,
        default="resources/data/embeddings",
        help="path to the dir where the video embeddings will be saved",
    )
    arg_parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="batch size with which frames will be processed by the clip",
    )
    return arg_parser


def main(args: Namespace):
    id2category = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with open(args.youtube_csv_file) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader, None)
        for row in tqdm(list(csv_reader)):
            id = row[0]
            category = row[-1]
            try:
                yt = YouTube(f"https://www.youtube.com/watch?v={id}")
                yt.streams.filter(
                    file_extension="mp4", resolution="360p"
                ).first().download(
                    output_path=args.save_videos_to, filename=f"{id}.mp4"
                )

                logger.info(f"Getting embeddings for the video {id}...")

                os.makedirs(args.save_videos_to, exist_ok=True)
                os.makedirs(args.save_embeddings_to, exist_ok=True)
                video_path = f"{args.save_videos_to}/{id}.mp4"
                embeddings_path = f"{args.save_embeddings_to}/{id}.json"

                frames = get_frames_from_video(video_path, args.frames_count)
                frames = list(map(preprocess, frames))
                frames_tensor = torch.stack(frames)
                batched_frames = frames_tensor.split(args.batch_size)

                with torch.no_grad():
                    all_embeddings = []
                    for batch in batched_frames:
                        batch = batch.to(device)
                        frame_features = model.encode_image(batch)
                        all_embeddings.append(frame_features.cpu().detach())

                    with open(embeddings_path, "w+") as embedding_file:
                        json.dump(torch.cat(all_embeddings).tolist(), embedding_file)
                id2category[id] = category
            except Exception as e:
                logger.error(f"Failed to download or get embeddings for video {id}: {str(e)}")
                continue

    categories = list(id2category.values())
    logger.info(f"Downloaded {len(categories)} videos")

    count = Counter(categories)
    logger.info(count)

    with open(args.save_config_to, "w+") as config_file:
        json.dump(id2category, config_file)


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
