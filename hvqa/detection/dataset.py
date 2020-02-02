import torch
import json
import numpy as np
from pathlib import Path
from PIL import Image

from hvqa.util import UnknownObjectTypeException


class _AbsDetectionDataset(torch.utils.data.Dataset):
    """
    Abstract class for finding the dataset for object detection
    """

    def __init__(self, data_dir, img_size, num_regions):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.num_regions = num_regions
        self.region_size = int(self.img_size / self.num_regions)

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

        output = torch.zeros([5, 8, 8], dtype=torch.float32)
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
                        # vec = torch.cat((normalised_coords, one, class_vec))
                        vec = torch.cat((normalised_coords, one))
                        output[:, j, i] = vec

        return output

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


class DetectionDataset(_AbsDetectionDataset):
    """
    A class for storing the dataset of frames
    Generates tensors for frames and outputs on the fly
    Note: For object detection we do not care which video the frames came from so all frames are stored together
    """

    def __init__(self, data_dir, img_size, num_regions):
        super().__init__(data_dir, img_size, num_regions)
        self.ids, self.frame_dicts = self._find_frames()

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

    def __getitem__(self, item):
        video_num, frame_num = self.ids[item]
        frame_dict = self.frame_dicts[item]
        video_dir = Path(f"{self.data_dir}/{video_num}")
        return self._collect_frame(video_dir, frame_num), self._collect_frame_output(frame_dict)

    def __len__(self):
        return len(self.ids)
