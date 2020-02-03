import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util import get_device, load_model, DTYPE, NUM_YOLO_REGIONS
from hvqa.detection.hyperparameters import *
from hvqa.detection.models import DetectionModel, ClassifierModel, DetectionBackbone
from hvqa.detection.dataset import DetectionDataset, ClassificationDataset


PRINT_BATCHES = 100

_mse_func = nn.MSELoss(reduction="none")


def calc_loss_detection(pred, actual):
    mse = _mse_func(pred, actual)
    mse = mse[:, 0:4, :, :] * COORD_MULT
    loss_sum = torch.sum(mse, [1, 2, 3])
    return torch.mean(loss_sum)


def calc_loss_classifiction(pred, actual):
    mse = _mse_func(pred, actual)
    loss_sum = torch.sum(mse, [1])
    return torch.mean(loss_sum)


def train_model(train_loader, model, optimiser, model_save, loss_func, scheduler=None, epochs=100):
    device = get_device()

    print(f"Training model using device: {device}")

    current_save = None
    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device, dtype=DTYPE)
            y = y.to(device=device, dtype=DTYPE)

            output = model(x)
            loss = loss_func(output, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t % PRINT_BATCHES == 0:
                print(f"Epoch {e:4d}, batch {t:4d} -- Loss = {loss.item():.6f}, lr = {optimiser.param_groups[0]['lr']:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Save a temp model every epoch
        current_save = f"{model_save}/after_{e+1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

    print(f"Completed training, final model saved to {current_save}")


def train_detector(train_dir, model_save_dir):
    dataset = DetectionDataset(train_dir, NUM_YOLO_REGIONS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = DetectionModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(loader, model, optimiser, model_save_dir, calc_loss_detection)


def train_classifier(train_dir, model_save_dir):
    dataset = ClassificationDataset(train_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ClassifierModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(loader, model, optimiser, model_save_dir, calc_loss_classifiction)


def main(train_dir, model_save_dir, classifier):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    if classifier:
        print("Training classification model...")
        train_classifier(train_dir, model_save_dir)
    else:
        print("Training detection model...")
        train_detector(train_dir, model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    parser.add_argument("-c", "--classifier", action="store_true", default=False)
    args = parser.parse_args()
    main(args.train_dir, args.model_save_dir, args.classifier)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("dir", type=str)
#     args = parser.parse_args()
#
#     pretrained = load_model(ClassifierModel, args.dir)
#     model = DetectionBackbone(pretrained)
#     img = torch.rand(64, 3, 128, 128)
#     pred = model(img)
#     print(pred.shape)
