import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util.func import get_device
from hvqa.spec.definitions import detector_transforms, DTYPE
from hvqa.detection.models import ClassifierModel
from hvqa.detection.dataset import ClassificationDataset
from hvqa.detection.evaluation import ClassificationEvaluator


BATCH_SIZE = 128
LEARNING_RATE = 0.001

_mse_func = nn.MSELoss(reduction="none")


def calc_loss_classifiction(pred, actual):
    mse = _mse_func(pred, actual)
    loss_sum = torch.sum(mse, [1])
    return torch.mean(loss_sum)


def train_one_epoch(model, optimiser, loader_train, device, epoch, print_freq=20):
    for t, (x, y) in enumerate(loader_train):
        model.train()
        x = x.to(device=device, dtype=DTYPE)
        y = y.to(device=device, dtype=DTYPE)

        output = model(x)
        loss = calc_loss_classifiction(output, y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if t % print_freq == 0:
            print(f"Epoch {epoch:>3}, batch {t:>4} "
                  f"-- loss = {loss.item():.6f} "
                  f"-- lr = {optimiser.param_groups[0]['lr']:.4f}")


def train_classifier(model, loader_train, loader_test, model_save_dir, epochs=50):
    evaluator = ClassificationEvaluator(loader_test)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = get_device()

    print(f"Training classification model using device {device}...")

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


def main(train_dir, test_dir, model_save_dir):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    model = ClassifierModel()

    # Read train data
    dataset_train = ClassificationDataset(train_dir, transforms=detector_transforms)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)

    # Read test data
    dataset_test = ClassificationDataset(test_dir, transforms=detector_transforms)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

    train_classifier(model, loader_train, loader_test, model_save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training detector backbone on classification task")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.model_save_dir)
