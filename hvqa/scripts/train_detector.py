import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from hvqa.detection.dataset import DetectionDataset
from hvqa.detection.model import DetectionModel


IMG_SIZE = 128
NUM_REGIONS = 8

BATCH_SIZE = 48
LEARNING_RATE = 0.001

USE_GPU = True
DTYPE = torch.float32

PRINT_BATCHES = 100


def build_data_loader(dataset_dir):
    dataset = DetectionDataset(dataset_dir, IMG_SIZE, NUM_REGIONS)
    num_samples = len(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(num_samples)))
    return loader


def train_model(train_loader, model, optimiser, model_save, scheduler=None, epochs=100):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"Training model using device: {device}")

    loss_func = nn.MSELoss()

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
                print(f"Epoch {e:4d}, batch {t:4d} -- Loss = {loss.item():.4f}, lr = {optimiser.param_groups[0]['lr']:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Save a temp model every epoch
        current_save = f"{model_save}/yolo_detection_model_after_{e}_epochs.pt"
        torch.save(model.state_dict(), current_save)

    print(f"Completed training, final model saved to {current_save}")


def main(train_dir, model_save_dir):
    loader = build_data_loader(train_dir)
    model = DetectionModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimiser, 1)

    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    train_model(loader, model, optimiser, model_save_dir, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.model_save_dir)
