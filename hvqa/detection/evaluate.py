import argparse
import torch
from torch.utils.data import DataLoader

from hvqa.spec.definitions import detector_transforms
from hvqa.util.func import collate_func, get_device
from hvqa.detection.evaluation import DetectionEvaluator
from hvqa.detection.models import DetectionModel, DetectionBackbone
from hvqa.detection.dataset import DetectionDataset


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"
DEFAULT_CONF_THRESHOLD = 0.5


def eval_detector(test_dir):
    print("Evaluating detector performance...")

    device = get_device()

    backbone = DetectionBackbone()
    model = DetectionModel(backbone)
    model.load_state_dict(torch.load(DETECTOR_PATH, map_location=device))
    model = model.to(device)

    dataset_test = DetectionDataset(test_dir, transforms=detector_transforms)
    loader_test = DataLoader(dataset_test, batch_size=128, shuffle=True, collate_fn=collate_func)

    evaluator = DetectionEvaluator(loader_test)
    evaluator.eval_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    args = parser.parse_args()
    eval_detector(args.eval_dir)
