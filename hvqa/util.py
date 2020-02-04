# File for helper classes and methods

import json
from pathlib import Path
import torch
import cv2
import numpy as np


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


# *** Util functions ***


def resize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * IMG_SIZE)
    y1 = round(y1 * IMG_SIZE)
    x2 = round(x2 * IMG_SIZE)
    y2 = round(y2 * IMG_SIZE)
    return x1, y1, x2, y2


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
    print(f"Loaded classification model with device: {device}")
    return model


def get_device():
    return torch.device("cuda:0") if USE_GPU and torch.cuda.is_available() else torch.device("cpu")


def extract_bbox_and_class(img_out, conf_threshold):
    preds = img_out[:, img_out[4, :, :] > conf_threshold]

    preds_arr = []
    for pred_idx in range(preds.shape[1]):
        pred = preds[:, pred_idx]
        bbox = resize_bbox(pred[0:4].numpy())
        conf, idx = torch.max(pred[5:], 0)
        preds_arr.append((bbox, conf.item(), idx.item()))

    return preds_arr


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
    edges = cv2.Canny(gray, IMG_MIN_VAL, IMG_MAX_VAL)[None, :, :]
    output = np.concatenate((edges, cv_img.transpose((2, 1, 0))), axis=0)
    return output
