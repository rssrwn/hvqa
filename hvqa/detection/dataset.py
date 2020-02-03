import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from hvqa.util import UnknownObjectTypeException


IMG_SIZE = 128


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

        print("Searching videos for training data...")

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

        print(f"Found training data from {num_videos} videos and {num_frames} frames")

        return ids, frame_dicts

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

    @staticmethod
    def _collect_frame(video_dir, frame_idx):
        """
        Produce a torch Tensor of an image, normalised between 0 and 1
        :param video_dir: Path object of directory image is stored in
        :param frame_idx: Frame index with video directory
        :return: Image as a torch Tensor (3 x height x width)
        """

        img_file = video_dir / f"frame_{frame_idx}.png"
        if img_file.exists():
            img = Image.open(img_file)
            img_arr = np.transpose(np.asarray(img, dtype=np.float32) / 255, (2, 0, 1))
            img_tensor = torch.from_numpy(img_arr)
            img.close()
        else:
            raise FileNotFoundError(f"Could not find image: {img_file}")

        return img_tensor

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


class DetectionDataset(_AbsHVQADataset):
    """
    A class for storing the dataset of frames
    Generates tensors for frames and outputs on the fly
    Note: For object detection we do not care which video the frames came from so all frames are stored together
    """

    def __init__(self, data_dir, num_regions, transforms=None):
        super(DetectionDataset, self).__init__(data_dir, transforms)
        self.num_regions = num_regions
        self.region_size = int(self.img_size / self.num_regions)

    def __getitem__(self, item):
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
        if self.transforms is not None:
            t = self.transforms
            img = t(img)

        return img, target

    def _collect_targets(self, frame_dict):
        objects = frame_dict["objects"]
        bboxs = torch.zeros((len(objects), 4), dtype=torch.float32)
        labels = torch.zeros(len(objects), dtype=torch.int64)
        areas = torch.zeros(len(objects), dtype=torch.float32)
        for idx, obj in enumerate(frame_dict["objects"]):
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
        if obj_type == "octopus":
            return 0
        elif obj_type == "fish":
            return 1
        elif obj_type == "bag":
            return 2
        elif obj_type == "rock":
            return 3
        else:
            raise UnknownObjectTypeException(f"Unknown object {obj}")

    def _collect_frame_output(self, frame_dict):
        """
        Build frame output for YOLO-style training
        For each region of the image we produce a vector with 9 numbers:
          - x1, y1, x2, y2, confidence, p(octo), p(fish), p(bag), p(rock)

        Probabilities and confidence are either 0 or 1
        Coords in output are normalised between 0 and 1

        :param frame_dict: Frame dictionary (as in json file of dataset)
        :return: Frame output as a 9x8x8 torch Tensor (Note: outputs x height x width)
        """

        output = torch.zeros([9, 8, 8], dtype=torch.float32)
        objects = frame_dict["objects"]
        for obj in objects:
            [x1, y1, x2, y2] = obj["position"]
            class_vec = self._create_class_vec(obj)
            normalised_coords = torch.from_numpy(np.array([x1, y1, x2, y2], dtype=np.float32) / self.img_size)

            centre_x = (x1 + x2) // 2
            centre_y = (y1 + y2) // 2

            # Check which region the centre of object is in
            for i in range(0, self.num_regions):
                for j in range(0, self.num_regions):
                    r_s = self.region_size
                    if (i * r_s) <= centre_x < (i + 1) * r_s and (j * r_s) <= centre_y < (j + 1) * r_s:
                        one = torch.ones([1], dtype=torch.float32)
                        vec = torch.cat((normalised_coords, one, class_vec))
                        output[:, j, i] = vec

        return output

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


class ClassificationDataset(_AbsHVQADataset):
    def __init__(self, data_dir):
        super(ClassificationDataset, self).__init__(data_dir)

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
        video_num, frame_num = self.ids[item]
        frame_dict = self.frame_dicts[item]
        video_dir = Path(f"{self.data_dir}/{video_num}")
        return self._collect_frame(video_dir, frame_num), self._collect_classifier_output(frame_dict)
