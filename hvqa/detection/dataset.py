import torch
import json
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from hvqa.util import UnknownObjectTypeException, collect_img, IMG_SIZE


class _AbsHVQADataset(Dataset):
    """
    Abstract class for datasets for both classification and detection tasks
    """

    def __init__(self, data_dir, transforms=None):
        super(_AbsHVQADataset, self).__init__()

        self.data_dir = data_dir
        self.img_size = IMG_SIZE
        self.transforms = transforms
        self.ids, self.frame_dicts = self._find_frames()

    def __getitem__(self, item):
        raise NotImplementedError("Cannot call this method on an abstract class")

    def __len__(self):
        return len(self.ids)

    def _find_frames(self):
        basepath = Path(self.data_dir)
        video_dirs = basepath.iterdir()

        ids = []
        frame_dicts = []

        num_videos = 0
        num_frames = 0

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
                    ids.append((video_num, frame_num))
                    frame_dicts.append(frame)
                    num_frames += 1

            num_videos += 1

        print(f"Found data from {num_videos} videos and {num_frames} frames")

        return ids, frame_dicts

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return collect_img(video_dir, frame_idx)

    def get_image(self, item):
        """
        Get a PIL Image of the frame and frame_dict for that image
        Note: The item is the number stored in the dataset, not the index in a video

        :param item: Frame number
        :return: PIL Image, frame_dict
        """

        video_num, frame_num = self.ids[item]
        img_path = Path(f"{self.data_dir}/{video_num}/frame_{frame_num}.png")
        json_file = Path(self.data_dir) / str(video_num) / "video.json"
        if json_file.exists():
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            frames = video_dict["frames"]
        else:
            raise FileNotFoundError(f"{json_file} does not exist")

        return Image.open(img_path), frames[frame_num]

    def _apply_trans(self, img):
        if self.transforms is not None:
            img = self.transforms(img)

        return img


class DetectionDataset(_AbsHVQADataset):
    """
    A class for storing the dataset of frames in the Faster R-CNN format
    Generates tensors for frames and outputs on the fly
    Note: For object detection we do not care which video the frames came from so all frames are stored together
    """

    def __init__(self, data_dir, transforms=None):
        super(DetectionDataset, self).__init__(data_dir, transforms)

    def __getitem__(self, item):
        """
        Get a training pair (input, target)

        :param item: Index of training data point
        :return: img (tensor), target (dict)
        """

        video_num, frame_num = self.ids[item]
        frame_dict = self.frame_dicts[item]
        video_dir = Path(f"{self.data_dir}/{video_num}")

        boxes, labels, areas, is_crowd = self._collect_targets(frame_dict)
        image_id = torch.tensor([item])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": areas,
            "iscrowd": is_crowd
        }

        img = self._collect_img(video_dir, frame_num)
        img = self._apply_trans(img)
        return img, target

    def _collect_targets(self, frame_dict):
        objects = frame_dict["objects"]
        bboxs = torch.zeros((len(objects), 4), dtype=torch.float32)
        labels = torch.zeros(len(objects), dtype=torch.int64)
        areas = torch.zeros(len(objects), dtype=torch.float32)
        for idx, obj in enumerate(objects):
            bbox = obj["position"]
            bboxs[idx, :] = torch.tensor(bbox)

            label = self._label(obj)
            labels[idx] = label

            area = self._calc_area(bbox)
            areas[idx] = area

        is_crowd = torch.zeros(len(objects), dtype=torch.uint8)
        return bboxs, labels, areas, is_crowd

    @staticmethod
    def _calc_area(bbox):
        x1, y1, x2, y2 = bbox
        x_diff = x2 - x1
        y_diff = y2 - y1
        return x_diff * y_diff

    @staticmethod
    def _label(obj):
        obj_type = obj["class"]

        # Note: 0 represents background
        if obj_type == "octopus":
            return 1
        elif obj_type == "fish":
            return 2
        elif obj_type == "bag":
            return 3
        elif obj_type == "rock":
            return 4
        else:
            raise UnknownObjectTypeException(f"Unknown object {obj}")


class ClassificationDataset(_AbsHVQADataset):
    def __init__(self, data_dir, transforms=None):
        super(ClassificationDataset, self).__init__(data_dir, transforms)

    @staticmethod
    def _collect_classifier_output(frame_dict):
        objs = {obj["class"] for obj in frame_dict["objects"]}
        output = torch.zeros(4, dtype=torch.float32)
        for obj in objs:
            if obj == "octopus":
                output[0] = 1.0
            elif obj == "fish":
                output[1] = 1.0
            elif obj == "bag":
                output[2] = 1.0
            elif obj == "rock":
                output[3] = 1.0
            else:
                raise UnknownObjectTypeException(f"Unknown object {obj}")

        return output

    def __getitem__(self, item):
        """
        Get training pair (input, target)

        :param item: Index of training data point
        :return: Img (img tensor), target (classification tensor)
        """

        video_num, frame_num = self.ids[item]
        frame_dict = self.frame_dicts[item]
        video_dir = Path(f"{self.data_dir}/{video_num}")

        target = self._collect_classifier_output(frame_dict)
        img = self._collect_img(video_dir, frame_num)
        img = self._apply_trans(img)
        return img, target
