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

        frames_arr = []
        outputs_arr = []
        for video_dir in video_dirs:
            json_file = video_dir / "video.json"
            if json_file.exists():
                with json_file.open() as f:
                    json_text = f.read()

                video_dict = json.loads(json_text)
                frames = video_dict["frames"]
                for i, frame in enumerate(frames):
                    outputs_arr.append(cls._collect_frame_output(frame))
                    frames_arr.append(cls._collect_frame(video_dir, i))

        frames_tensor = torch.stack(frames_arr)
        outputs_tensor = torch.stack(outputs_arr)
        return frames_tensor, outputs_tensor

    @staticmethod
    def _collect_frame_output(frame):
        pass

    @staticmethod
    def _collect_frame(video_dir, frame_idx):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
