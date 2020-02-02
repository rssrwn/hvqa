import argparse
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

from hvqa.util import *
from hvqa.detection.model import DetectionModel


CONF_THRESHOLD = 0.5


def _add_bboxs(drawer, positions, ground_truth=True):
    colour = "blue" if ground_truth else "red"
    for position in positions:
        x1, y1, x2, y2 = position
        x1 -= 1
        y1 -= 1
        x2 += 1
        y2 += 1
        drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour)


def visualise(data_path, model_path=None):
    json_file = data_path / "video.json"
    if json_file.exists():
        with json_file.open() as f:
            json_text = f.read()

        video_dict = json.loads(json_text)

    else:
        raise FileNotFoundError(f"{json_file} does not exist")

    num_frames = len(video_dict["frames"])
    frame_idx = random.randint(0, num_frames - 1)

    # Collect image
    img_path = data_path / f"frame_{frame_idx}.png"
    img = Image.open(img_path)

    # Add bboxs
    draw = ImageDraw.Draw(img)
    frame_dict = video_dict["frames"][frame_idx]
    _add_bboxs(draw, [obj["position"] for obj in frame_dict["objects"]])

    if model_path:
        model = load_model(DetectionModel, model_path)
        img_arr = np.transpose(np.asarray(img, dtype=np.float32) / 255, (2, 0, 1))
        img_tensor = torch.from_numpy(img_arr)

        with torch.no_grad():
            net_out = model(img_tensor[None, :, :, :])

        preds = extract_bbox_and_class(net_out[0, :, :, :], CONF_THRESHOLD)
        _add_bboxs(draw, [pred[0] for pred in preds], ground_truth=False)

    img.show()


def main(data_dir, model_file=None):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} does not exist")

    model_path = None
    if model_file:
        model_path = Path(model_file)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist")

    visualise(data_path, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for visualising bounding boxes")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.model)
