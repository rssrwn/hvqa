import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset

import hvqa.util.func as util


class BaselineDataset(Dataset):
    def __init__(self, ids, frames, questions, q_types, answers):
        ids = sorted(enumerate(ids), key=lambda idx_id: idx_id[1])
        self.ids = ids
        self.frames = [frames[idx] for idx, _ in ids]
        self.questions = [questions[idx] for idx, _ in ids]
        self.q_types = [q_types[idx] for idx, _ in ids]
        self.answers = [answers[idx] for idx, _ in ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        frames = self.frames[item]
        questions = self.questions[item]
        q_types = self.q_types[item]
        answers = self.answers[item]
        return frames, questions, q_types, answers

    @classmethod
    def from_data_dir(cls, data_dir):
        data_dir = Path(data_dir)
        videos = cls._collect_videos(data_dir)
        ids, video_dicts, frames = tuple(zip(*videos))

        questions = []
        q_types = []
        answers = []
        for video_dict in video_dicts:
            questions.append(video_dict["questions"])
            q_types.append(video_dict["question_types"])
            answers.append(video_dict["answers"])

        dataset = BaselineDataset(ids, frames, questions, q_types, answers)
        return dataset

    @classmethod
    def _collect_videos(cls, path):
        """
        Collect all videos under <path> as list of PIL images, ids and dicts
        This function executes asynchronously

        :param path: Path obj
        :return: [(id, dict, [PIL Image])]
        """

        num_workers = os.cpu_count()
        future_timeout = 5
        executor = ThreadPoolExecutor(max_workers=num_workers)

        futures = []
        for video_dir in path.iterdir():
            video_num = str(video_dir).split("/")[-1]
            if not video_num.isdigit():
                print(f"WARNING: {video_dir} could not be parsed into an integer")
                continue

            future = executor.submit(cls._collect_video, video_dir)
            futures.append((video_num, future))

        videos = []
        for video_num, future in futures:
            video_num = int(video_num)
            video_dict, imgs = future.result(future_timeout)
            videos.append((video_num, video_dict, imgs))

        return videos

    @classmethod
    def _collect_video(cls, video_dir):
        json_file = video_dir / "video.json"
        with json_file.open() as f:
            json_text = f.read()

        video_dict = json.loads(json_text)
        num_frames = len(video_dict["frames"])
        images = [cls._collect_img(video_dir, frame) for frame in range(num_frames)]

        return video_dict, images

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)
