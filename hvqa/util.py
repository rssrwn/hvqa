# File for helper classes and methods

import json
from pathlib import Path
from PIL import Image
import torch
import cv2
import numpy as np
import torchvision.transforms as T


# *** Classes ***

class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


# *** Exceptions ***

class UnknownObjectTypeException(BaseException):
    pass


class UnknownPropertyException(BaseException):
    pass


# *** Definitions ***

IMG_SIZE = 128
NUM_YOLO_REGIONS = 8

USE_GPU = True
DTYPE = torch.float32

PROPERTIES = ["colour", "rotation", "class"]
COLOURS = ["red", "silver", "white", "brown", "blue", "purple", "green"]
ROTATIONS = [0, 1, 2, 3]
CLASSES = ["octopus", "fish", "bag", "rock"]
PROPS_ARR = [COLOURS, ROTATIONS, CLASSES]

# Means:    (0.010761048, 0.24837227, 0.75161874, 0.6989449)
# Std devs: (0.10317583, 0.06499335, 0.065463744, 0.056644086)

detector_transforms = T.Compose([
    T.ToTensor(),
])

prop_transforms = T.Compose([
    T.Resize((16, 16)),
    T.ToTensor(),
])


# *** Util functions ***


def load_model(model_class, path, *model_args):
    """
    Load a model whose state_dict has been saved
    Note: This does work for models whose entire object has been saved

    :param model_class: Class of model to load
    :param path: Path to model params
    :param model_args: Args to pass to model init
    :return: Model object
    """

    device = get_device()
    model = model_class(*model_args)
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Loaded model with device: {device}")
    return model


def get_device():
    return torch.device("cuda:0") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")


def get_video_dicts(data_dir):
    directory = Path(data_dir)

    dicts = []
    num_dicts = 0
    for video_dir in directory.iterdir():
        json_file = video_dir / "video.json"
        if json_file.exists():
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            dicts.append(video_dict)
            num_dicts += 1

        else:
            raise FileNotFoundError(f"{json_file} does not exist")

    print(f"Successfully extracted {num_dicts} video dictionaries from json files")
    return dicts


def collect_img(video_dir, frame_idx):
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


def collect_obj(img, position):
    """
    Collect an object from its bbox in an image

    :param img: PIL Image
    :param position: bbox coords
    :return: Cropped PIL Image
    """

    # Add a 1x1 border around object
    x1, y1, x2, y2 = position
    x1 -= 1
    y1 -= 1
    x2 += 2
    y2 += 2
    return img.crop((x1, y1, x2, y2))


IMG_MIN_VAL = 0
IMG_MAX_VAL = 0


def add_edges(img):
    """
    Add an edge channel to the image

    :param img: PIL Image
    :return: Numpy array of dimension (C + 1, H, W), channel 0 are edges (between 0 and 255)
    """

    cv_img = np.array(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, IMG_MIN_VAL, IMG_MAX_VAL)[:, :, None]
    output = np.concatenate((edges, cv_img), axis=2)
    return output


def add_bboxs(drawer, positions, colour):
    for position in positions:
        x1, y1, x2, y2 = position
        x1 = round(x1) - 1
        y1 = round(y1) - 1
        x2 = round(x2) + 1
        y2 = round(y2) + 1
        drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour)


def collate_func(batch):
    return tuple(zip(*batch))
