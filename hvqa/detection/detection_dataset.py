import torch
import json
from pathlib import Path


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        pass

    @classmethod
    def _collect_frames(cls, data_dir):
        basepath = Path(data_dir)
        video_dirs = basepath.iterdir()

        print("Writing detection outputs for frames...")

        num_videos = 0
        for video_dir in video_dirs:
            json_file = video_dir / "video.json"
            if json_file.exists():
                with json_file.open() as f:
                    json_text = f.read()

                video_dict = json.loads(json_text)
                frames = video_dict["frames"]

                outputs = []
                for frame in frames:
                    output_tensor = cls._collect_frame_output(frame)

    @classmethod
    def _collect_frame_output(cls, frame):
        return 3

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
