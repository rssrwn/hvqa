import argparse
import json
from pathlib import Path

from hvqa.util import *
from hvqa.detection.dataset import DetectionBatchDataset
from hvqa.detection.model import DetectionModel


def evaluate_models(loader, models, names):
    for idx, model in enumerate(models):
        name = names[idx]
        model.eval()

        for _, (x, y) in enumerate(loader):
            pred = model(x)

    pass


def load_models(model_path):
    models = []
    files = []

    # Allow directories
    if model_path.is_dir():
        for model_file in model_path.iterdir():
            models.append(load_obj_detection_model(model_file))
            files.append(model_file)

    elif model_path.exists():
        models.append(load_obj_detection_model(model_path))
        files.append(model_path)

    else:
        raise FileNotFoundError(f"Either {model_path} does not exist or does not contain any model files")

    return models, files


def load_obj_detection_model(path):
    """
    Load a model whose state_dict has been saved
    Note: This does work for models whose entire object has been saved

    :param path: Path to model params
    :return: Model object
    """

    model = DetectionModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def main(test_dir, model_file):
    test_path = Path(test_dir)
    model_path = Path(model_file)
    models, names = load_models(model_path)
    test_loader = build_data_loader(DetectionBatchDataset, test_path, 48)
    evaluate_models(test_loader, models, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    main(args.eval_dir, args.model_path)
