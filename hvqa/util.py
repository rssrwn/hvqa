# File for helper classes and methods

import json
from pathlib import Path
import torch
from torch.utils.data.dataloader import DataLoader


# *** Exceptions ***

class UnknownObjectTypeException(BaseException):
    pass


class UnknownPropertyException(BaseException):
    pass


# *** Definitions ***

IMG_SIZE = 128
NUM_REGIONS = 8

USE_GPU = True
DTYPE = torch.float32


# *** Util functions ***

# TODO Remove
def build_data_loader(dataset_class, dataset_dir, batch_size):
    """
    Builds a DataLoader object from given dataset

    :param dataset_class: Class of dataset to build, must take three init params: dir, img_size and num_regions
    :param dataset_dir: Path object of dataset directory (stored in hvqa format)
    :param batch_size: Size of mini-batches to read from dataset
    :return: DataLoader object which iterates over the dataset
    """

    dataset = dataset_class(dataset_dir, IMG_SIZE, NUM_REGIONS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def resize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * IMG_SIZE)
    y1 = round(y1 * IMG_SIZE)
    x2 = round(x2 * IMG_SIZE)
    y2 = round(y2 * IMG_SIZE)
    return x1, y1, x2, y2


def load_model(model_class, path):
    """
    Load a model whose state_dict has been saved
    Note: This does work for models whose entire object has been saved

    :param model_class: Class of model to load
    :param path: Path to model params
    :return: Model object
    """

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = model_class()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


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
