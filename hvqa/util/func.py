# *** util functions ***

import torch
import json
from pathlib import Path
from PIL import Image


_USE_GPU = True


def inc_in_map(coll, key):
    val = coll.get(key)
    if val is None:
        coll[key] = 0

    coll[key] += 1


def append_in_map(coll, key, item):
    items = coll.get(key)
    items = [] if items is None else items
    items.append(item)
    coll[key] = items


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


def obj_encoding(spec, obj):
    obj_enc = list(map(lambda v: 1.0 if v == obj.cls else 0.0, spec.obj_types()))
    obj_position = list(map(lambda p: p / 255, obj.pos))
    obj_enc.extend(obj_position)
    for prop, val in obj.prop_vals.items():
        prop_enc = property_encoding(spec, prop, val)
        obj_enc.extend(prop_enc)

    return obj_enc


def property_encoding(spec, prop, val):
    vals = spec.prop_values(prop)
    one_hot = list(map(lambda v: 1.0 if v == val else 0.0, vals))
    assert sum(one_hot) == 1.0, f"Val {val} is not in property values {vals}"
    return one_hot


def encode_obj_vector(spec, obj, obj_feat, tensor_pos=False):
    cls = [0.0] * 4
    val_idx = spec.obj_types().index(obj.cls)
    cls[val_idx] = 1.0

    obj_id = [0.0] * 20
    if obj.id < 20:
        obj_id[obj.id] = 1.0

    obj_feat = list(obj_feat) + cls + obj_id
    if tensor_pos:
        obj_pos = [pos / 255 for pos in obj.pos]
        obj_ = torch.tensor(obj_feat + obj_pos)
    else:
        obj_ = (torch.tensor(obj_feat), obj.pos)

    return obj_
