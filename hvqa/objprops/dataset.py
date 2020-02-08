import json
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset


COLOURS = ["red", "silver", "white", "brown", "blue", "purple", "green"]
ROTATIONS = [0, 1, 2, 3]


class PropertyExtractionDataset(Dataset):
    """
    Dataset for training/testing networks which extract properties from objects
    """

    def __int__(self, data_dir, transforms=None):
        super(PropertyExtractionDataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.ids, self.obj_dicts = self._find_objects()

    def __getitem__(self, item):
        vid_idx, frame_idx, obj_idx = self.ids[item]
        obj_dict = self.obj_dicts[item]

        video_dir = self.data_dir / vid_idx
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
        """
        Collect an object from its bbox in an image

        :param img: PIL Image
        :param position: bbox coords
        :return: Cropped PIL Image
        """

        x1, y1, x2, y2 = position
        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
        return img.crop((x1, y1, x2, y2))

    @staticmethod
    def _collect_target(obj_dict):
        colour = obj_dict["colour"]
        colour_idx = COLOURS.index(colour)
        colour_tensor = torch.zeros(len(COLOURS))
        colour_tensor[colour_idx] = 1

        rotation = obj_dict["rotation"]
        rotation_idx = ROTATIONS.index(rotation)
        rotation_tensor = torch.zeros(len(ROTATIONS))
        rotation_tensor[rotation_idx] = 1

        target = {
            "colour": colour_tensor,
            "rotation": rotation_tensor
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
                    objects = frame["objects"]

                    # Iterate through objects in frame
                    for obj_idx, obj in enumerate(objects):
                        obj_ids.append((video_num, frame_num, obj_idx))
                        objects.append(obj)
                        num_objects += 1

                    num_frames += 1
            num_videos += 1

        print(f"Found {num_objects} objects from {num_videos} videos and {num_frames} frames")

        return obj_ids, objects

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        """
        Produce a PIL image

        :param video_dir: Path object of directory image is stored in
        :param frame_idx: Frame index with video directory
        :return: PIL image
        """

        img_path = video_dir / f"frame_{frame_idx}.png"
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
        else:
            raise FileNotFoundError(f"Could not find image: {img_path}")

        return img
