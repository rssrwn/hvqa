import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from hvqa.detection.dataset import DetectionDataset
from hvqa.detection.model import DetectionModel


IMG_SIZE = 128
NUM_REGIONS = 8

BATCH_SIZE = 64
LEARNING_RATE = 0.001

USE_GPU = True
DTYPE = torch.float32

PRINT_BATCHES = 50

w_coord = 5
w_noobj = 0.5


def build_data_loader(dataset_dir):
    dataset = DetectionDataset(dataset_dir, IMG_SIZE, NUM_REGIONS)

    img, output = dataset[0]
    print(img)
    print(output)

    num_samples = len(dataset)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(range(num_samples)))
    return loader


def train_model(train_loader, model, optimiser, scheduler=None, epochs=100):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print(f"Training model using device {device}")

    loss = nn.MSELoss()

    model = model.to(device=device)
    for e in range(epochs):
        for t, (x, y) in enumerate(train_loader):
            model.train()
            x = x.to(device=device, dtype=DTYPE)
            y = y.to(device=device, dtype=torch.long)

            output = model(x)
            loss = loss(output, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t % PRINT_BATCHES == 0:
                print(f"Iteration {t}, loss = {loss.item():.4f}, lr = {optimiser.param_groups[0]['lr']:.4f}")

        if scheduler is not None:
            scheduler.step()


def main(train_dir):
    loader = build_data_loader(train_dir)
    model = DetectionModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimiser, 1)
    train_model(loader, model, optimiser, scheduler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir)
