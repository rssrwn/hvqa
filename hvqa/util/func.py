# *** util functions ***

import torch
import json
from pathlib import Path
from PIL import Image

from hvqa.util.exceptions import UnknownPropertyValueException


_USE_GPU = True


def inc_in_map(coll, key):
    val = coll.get(key)
    if val is None:
        coll[key] = 0

    coll[key] += 1


def get_or_default(coll, meta_data, key):
    default = meta_data[key]
    val = coll.get(key)
    val = val if val is not None else default
    return val


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
    model = model.to(device)
    print(f"Loaded model with device: {device}")
    return model


def save_model(model, path):
    """
    Save <model> to <path> using torch.save

    :param model: Model to save (nn.Module)
    :param path: Path to save to (str)
    """

    torch.save(model.state_dict(), path)


def get_device():
    return torch.device("cuda:0") if _USE_GPU and torch.cuda.is_available() else torch.device("cpu")


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


def format_prop_val(prop, prop_val):
    if prop == "rotation":
        if prop_val == "upward-facing":
            prop_val = 0
        elif prop_val == "right-facing":
            prop_val = 1
        elif prop_val == "downward-facing":
            prop_val = 2
        elif prop_val == "left-facing":
            prop_val = 3
        else:
            raise UnknownPropertyValueException(f"Rotation {prop_val} unknown")

    return prop_val


def format_prop_str(prop, prop_val):
    if prop == "rotation":
        if prop_val == 0 or prop_val == "0":
            prop_val = "upward-facing"
        elif prop_val == 1 or prop_val == "1":
            prop_val = "right-facing"
        elif prop_val == 2 or prop_val == "2":
            prop_val = "downward-facing"
        elif prop_val == 3 or prop_val == "3":
            prop_val = "left-facing"
        else:
            raise UnknownPropertyValueException(f"Rotation {prop} unknown")

    return prop_val


def event_to_asp_str(event):
    if event == "move" or event == "moving":
        asp_str = "move"
    elif event == "rotate left" or event == "rotating left":
        asp_str = "rotate_left"
    elif event == "rotate right" or event == "rotating right":
        asp_str = "rotate_right"
    elif event == "change colour" or event == "changing colour":
        asp_str = "change_colour"
    elif event == "eat a fish" or event == "eating a fish":
        asp_str = "eat_fish"
    elif event == "eat a bag" or event == "eating a bag":
        asp_str = "eat_bag"
    else:
        raise UnknownPropertyValueException(f"Unknown event {event}")

    return asp_str


def asp_str_to_event(asp_event):
    if asp_event == "move":
        event = "move"
    elif asp_event == "rotate_left":
        event = "rotate left"
    elif asp_event == "rotate_right":
        event = "rotate right"
    elif asp_event == "change_colour":
        event = "change colour"
    elif asp_event == "eat_fish":
        event = "eat a fish"
    elif asp_event == "eat_bag":
        event = "eat a bag"
    elif asp_event == "unknown":
        event = "unknown"
    else:
        raise UnknownPropertyValueException(f"Unknown ASP event {asp_event}")

    return event


# TODO Fix firth
occurrences = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "firth": 5
}


def format_occ(occ_str):
    occ = occurrences.get(occ_str)
    if occ is None:
        raise UnknownPropertyValueException(f"Unknown occurrence value {occ_str}")

    return occ
