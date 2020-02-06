import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from hvqa.util import get_device, add_edges, DTYPE
from hvqa.detection.hyperparameters import *
from hvqa.detection.models import ClassifierModel
from hvqa.detection.dataset import ClassificationDataset


PRINT_BATCHES = 50

_mse_func = nn.MSELoss(reduction="none")

detector_transforms = T.Compose([
    # T.Lambda(lambda x: add_edges(x)),
    T.ToTensor(),
])


def calc_loss_classifiction(pred, actual):
    mse = _mse_func(pred, actual)
    loss_sum = torch.sum(mse, [1])
    return torch.mean(loss_sum)


def train_classifier(train_loader, model, optimiser, model_save, loss_func, scheduler=None, epochs=100):
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
                print(f"Epoch {e:>3}, batch {t:>4} "
                      f"-- loss = {loss.item():.6f} "
                      f"-- lr = {optimiser.param_groups[0]['lr']:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Save a temp model every epoch
        current_save = f"{model_save}/after_{e+1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

    print(f"Completed training, final model saved to {current_save}")


def main(train_dir, model_save_dir):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    print("Training classification model...")
    dataset = ClassificationDataset(train_dir, transforms=detector_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ClassifierModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_classifier(loader, model, optimiser, model_save_dir, calc_loss_classifiction)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training detector backbone on classification task")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.model_save_dir)
