import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util import get_device, load_model, add_edges, detector_transforms, collate_func
from hvqa.detection.models import DetectionModel, ClassifierModel, DetectionBackboneWrapper
from hvqa.detection.dataset import DetectionDataset
from hvqa.detection.evaluation import DetectionEvaluator

from lib.vision.engine import train_one_epoch


BATCH_SIZE = 128
LEARNING_RATE = 0.005

_mse_func = nn.MSELoss(reduction="none")


def train_detector(model, loader_train, loader_test, model_save_dir, epochs=50):
    evaluator = DetectionEvaluator(loader_test)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = get_device()

    print(f"Training detection model using device {device}...")

    model = model.to(device=device)
    for epoch in range(epochs):
        model.train()
        train_one_epoch(model, optimiser, loader_train, device, epoch, print_freq=50)

        # Save a temp model every epoch
        current_save = f"{model_save_dir}/after_{epoch + 1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

        # Evaluate performance every epoch
        model.eval()
        evaluator.eval_model(model)

    print(f"Completed training, final model saved to {current_save}")


def main(train_dir, backbone_dir, val_dir, model_save_dir):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    # Read backbone model
    pretrained = load_model(ClassifierModel, backbone_dir)
    backbone = DetectionBackboneWrapper(pretrained)
    model = DetectionModel(backbone)

    # Read train data
    dataset_train = DetectionDataset(train_dir, transforms=detector_transforms)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    # Read test data
    dataset_test = DetectionDataset(val_dir, transforms=detector_transforms)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    train_detector(model, loader_train, loader_test, model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("backbone_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.backbone_dir, args.test_dir, args.model_save_dir)
