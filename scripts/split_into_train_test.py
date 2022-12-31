import json
import os
import random
from argparse import ArgumentParser, Namespace


def configure_arg_parser() -> ArgumentParser:
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--config-path",
        type=str,
        default="resources/data/config.json",
        help="path to json config with mapping video -> category",
    )
    arg_parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="share of the test sample from the entire dataset",
    )
    return arg_parser


def main(args: Namespace):
    with open(args.config_path, "r") as config_file:
        id2category = json.load(config_file)

    video_ids = list(id2category.keys())
    random.shuffle(video_ids)

    video_count = len(video_ids)
    train_size = int((1 - args.test_size) * video_count)
    train_ids, test_ids = video_ids[:train_size], video_ids[train_size:]

    train = [(id, id2category[id]) for id in train_ids]
    test = [(id, id2category[id]) for id in test_ids]

    dir = os.path.dirname(args.config_path)
    with open(f"{dir}/train_config.json", "w") as train_config_file:
        json.dump(train, train_config_file)
    with open(f"{dir}/test_config.json", "w") as test_config_file:
        json.dump(test, test_config_file)
    return train, test


if __name__ == "__main__":
    _args = configure_arg_parser().parse_args()
    main(_args)
