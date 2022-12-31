# 📺 Video Classification

Different approaches to video classification on [Youtube Videos Dataset](https://www.kaggle.com/datasets/rajatrc1705/youtube-videos-dataset) using [CLIP](https://openai.com/blog/clip/) embeddings for frames.

## Structure
* [`datasets`](./datasets) ‒ implementations of torch datasets (to get video embeddings based on multiple frames, to get a random video frame).
* [`models`](./models) ‒ models implementations.
* [`scripts`](./scripts) ‒ scripts for preparing data to training and evaluation.
* [`video_classification_utils`](./video_classification_utils) ‒ various useful utilities, e.g. for obtaining frames from video, for training models.
* [`experiments.ipynb`](./experiments.ipynb) ‒ notebook with running experiments, plots and metrics
## Requirements

Create virtual environment with `venv` or `conda` and install requirements:

```shell
pip install -r requirements.txt
```

Or build and run docker container:
```shell
./run_docker.sh
```

## Data

The [Youtube Videos Dataset](https://www.kaggle.com/datasets/rajatrc1705/youtube-videos-dataset) was used for training and testing.
The dataset contains information about the video and its subject matter. Our task involves classifying videos by subject using frame embeddings.

There are 4 video categories in total: _food_, _art_music_, _travel_, _history_.

#### Prepare embeddings
Videos were downloaded using the script [`prepare_embeddings.py`](./scripts/prepare_embeddings.py).
Since the dataset contains videos that have already been deleted, only those videos for which there is a valid link were downloaded.

A total of _1728_ videos were downloaded. 
* travel: 602
* food: 491
* art_music: 317
* history: 316

As the script name suggests, the script also extracts embeddings from videos.

![](./resources/images/get_embeddings.png)

The script extracts `--frames-count` (script parameter) frames from each video. Frames are extracted evenly, that is, the distance between all frames is the same.
Further, for each frame, the embedding vector is extracted using the [CLIP](https://github.com/openai/CLIP) model (ViT-B/32). The embedding sequence is saved to a json file in the directory `--save-embeddings-to`.

For further experiments, _100_ frames were extracted from each video.

#### Split into train test

Splitting into training and test sets is done using a script [`split_into_train_test.py`](./scripts/split_into_train_test.py). The share of the test sample from the total dataset is controlled by a parameter `--test-size` (default: _0.1_).

The split data statistics are as follows:

* Train (_1553_):
  * travel: 540
  * food: 442
  * history: 286
  * art_music: 285
* Test (_173_):
  * travel: 62
  * food: 49
  * art_music: 32
  * history: 30

## Models

## Results

