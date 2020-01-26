import torch
import json
import numpy as np
from pathlib import Path

from hvqa.util import *


IMAGE_SIZE = 128
NUM_REGIONS = 8
REGION_SIZE = int(IMAGE_SIZE / NUM_REGIONS)


class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        pass

    @staticmethod
    def _collect_frames(data_dir):
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
                    outputs_arr.append(DetectionDataset._collect_frame_output(frame))
                    frames_arr.append(DetectionDataset._collect_frame(video_dir, i))

        frames_tensor = torch.stack(frames_arr)
        outputs_tensor = torch.stack(outputs_arr)
        return frames_tensor, outputs_tensor

    @staticmethod
    def _collect_frame_output(frame):
        output = torch.zeros([8, 8, 9], dtype=torch.float32)
        objects = frame["objects"]
        for obj in objects:
            [x1, y1, x2, y2] = obj["position"]
            class_vec = DetectionDataset._create_class_vec(obj)
            centre_x = (x1 + x2) // 2
            centre_y = (y1 + y2) // 2
            normalised_coords = torch.from_numpy(np.array([x1, y1, x2, y2], dtype=np.float32) / IMAGE_SIZE)
            for i in range(0, NUM_REGIONS):
                for j in range(0, NUM_REGIONS):
                    if (i * REGION_SIZE) <= centre_x < (i + 1) * REGION_SIZE and \
                            (j * REGION_SIZE) <= centre_y < (j + 1) * REGION_SIZE:
                        one = torch.ones([1], dtype=torch.float32)
                        vec = torch.cat((normalised_coords, one, class_vec))
                        output[i, j] = vec

        return output

    @staticmethod
    def _collect_frame(video_dir, frame_idx):
        pass

    @staticmethod
    def _create_class_vec(obj):
        vec = torch.zeros([4], dtype=torch.float32)
        obj_type = obj["class"]
        if obj_type == "octopus":
            vec[0] = 1
        elif obj_type == "fish":
            vec[1] = 1
        elif obj_type == "bag":
            vec[2] = 1
        elif obj_type == "rock":
            vec[3] = 1
        else:
            raise UnknownObjectTypeException(f"Unknown object type: {obj_type}")

        return vec

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
