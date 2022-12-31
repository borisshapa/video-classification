import cv2
from PIL import Image
from tqdm import tqdm

HOUR_IN_SECONDS = 3600


def get_frames_from_video(video_path: str, n_frames: int):
    video_capture = cv2.VideoCapture(video_path)
    frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT) or video_capture.get(
        cv2.cv.CV_CAP_PROP_FRAME_COUNT
    )

    step_size_in_frames = frame_count // n_frames
    if step_size_in_frames < 1:
        step_size_in_frames = 1
        pbar = tqdm(total=frame_count)
    else:
        pbar = tqdm(total=n_frames)

    current_frame = 0
    frames = []
    while len(frames) < n_frames:
        success, frame = video_capture.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb_frame).convert("RGB"))
        current_frame += step_size_in_frames
        video_capture.set(1, current_frame)
        pbar.update(1)

    pbar.close()
    video_capture.release()
    return frames
