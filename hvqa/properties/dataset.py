import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

import hvqa.util.func as util


COLOURS = ["red", "silver", "white", "brown", "blue", "purple", "green"]
ROTATIONS = [0, 1, 2, 3]
CLASSES = ["octopus", "fish", "bag", "rock"]


class PropertyExtractionDataset(Dataset):
    """
    Dataset for training/testing networks which extract properties from objects
    """

    def __init__(self, data_dir, transforms=None):
        super(PropertyExtractionDataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.ids, self.obj_dicts = self._find_objects()

    def __getitem__(self, item):
        vid_idx, frame_idx, obj_idx = self.ids[item]
        obj_dict = self.obj_dicts[item]

        video_dir = self.data_dir / str(vid_idx)
        img = self._collect_img(video_dir, frame_idx)

        # Collect network input
        position = obj_dict["position"]
        obj = self._collect_obj(img, position)
        if self.transforms is not None:
            obj = self.transforms(obj)

        # Collect network output
        target = self._collect_target(obj_dict)

        return obj, target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _collect_obj(img, position):
        return util.collect_obj(img, position)

    @staticmethod
    def _collect_target(obj_dict):
        """
        Collect network target as a dictionary containing three targets

        :param obj_dict: Object
        :return: Dict containing single-elem tensors which are the idx of the output class
        """

        colour = obj_dict["colour"]
        colour_idx = COLOURS.index(colour)
        colour_tensor = torch.tensor([colour_idx])

        rotation = obj_dict["rotation"]
        rotation_idx = ROTATIONS.index(rotation)
        rotation_tensor = torch.tensor([rotation_idx])

        cls = obj_dict["class"]
        cls_idx = CLASSES.index(cls)
        cls_tensor = torch.tensor([cls_idx])

        target = {
            "colour": colour_tensor,
            "rotation": rotation_tensor,
            "class": cls_tensor
        }
        return target

    def _find_objects(self):
        video_dirs = self.data_dir.iterdir()

        obj_ids = []
        objects = []

        num_videos = 0
        num_frames = 0
        num_objects = 0

        print("Searching videos for data...")

        # Iterate through videos
        for video_dir in video_dirs:
            video_num = int(str(video_dir).split("/")[-1])
            json_file = video_dir / "video.json"
            if json_file.exists():
                with json_file.open() as f:
                    json_text = f.read()

                video_dict = json.loads(json_text)
                frames = video_dict["frames"]

                # Iterate through frames in current video
                for frame_num, frame in enumerate(frames):
                    objs = frame["objects"]

                    # Iterate through objects in frame
                    for obj_idx, obj in enumerate(objs):
                        obj_ids.append((video_num, frame_num, obj_idx))
                        objects.append(obj)
                        num_objects += 1

                    num_frames += 1
            num_videos += 1

        print(f"Found {num_objects} objects from {num_videos} videos and {num_frames} frames")

        return obj_ids, objects

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)