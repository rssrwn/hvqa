import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from hvqa.util import get_device, load_model, add_edges, DTYPE, NUM_YOLO_REGIONS
from hvqa.detection.hyperparameters import *
from hvqa.detection.models import DetectionModel, ClassifierModel, DetectionBackbone
from hvqa.detection.dataset import DetectionDataset, ClassificationDataset

from lib.vision.engine import train_one_epoch, evaluate


PRINT_BATCHES = 10

_mse_func = nn.MSELoss(reduction="none")


detector_transforms = T.Compose([
    # T.Lambda(lambda x: add_edges(x)),
    T.ToTensor(),
])


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


def train_detector_loop(train_loader, model, optimiser, model_save, scheduler=None, epochs=100):
    device = get_device()

    print(f"Training model using device: {device}")

    current_save = None
    model = model.to(device=device)
    for e in range(epochs):
        for t, (images, targets) in enumerate(train_loader):
            model.train()

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = torch.tensor([loss * REG_MULT if key == "loss_box_reg" else loss for key, loss in loss_dict.items()], requires_grad=True)
            losses[1] *= REG_MULT
            loss = torch.sum(losses)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t % PRINT_BATCHES == 0:
                print(f"Epoch [{e}], batch [{t}] -- "
                      f"loss_classifier {losses[0]:.4f}, loss_box_reg {losses[1]:.4f}, "
                      f"loss_objectness {losses[2]:.4f}, loss_rpn_box_reg {losses[3]:.4f} -- "
                      f"lr {optimiser.param_groups[0]['lr']:.4f}")

        if scheduler is not None:
            scheduler.step()

        # Save a temp model every epoch
        current_save = f"{model_save}/after_{e + 1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

    print(f"Completed training, final model saved to {current_save}")


def collate_func(batch):
    return tuple(zip(*batch))


def train_detector(train_dir, model_save_dir, backbone_dir):
    device = get_device()
    dataset = DetectionDataset(train_dir, NUM_YOLO_REGIONS, transforms=detector_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)
    pretrained = load_model(ClassifierModel, backbone_dir)
    backbone = DetectionBackbone(pretrained)
    model = DetectionModel(backbone)
    model = model.to(device=device)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_detector_loop(loader, model, optimiser, model_save_dir)

    # num_epochs = 10
    # for epoch in range(num_epochs):
    #     train_one_epoch(model, optimiser, loader, device, epoch, print_freq=2)
    #     current_save = f"{model_save_dir}/after_{epoch + 1}_epochs.pt"
    #     torch.save(model.state_dict(), current_save)


def train_classifier(train_dir, model_save_dir):
    dataset = ClassificationDataset(train_dir, transforms=detector_transforms)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = ClassifierModel()
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_model(loader, model, optimiser, model_save_dir, calc_loss_classifiction)


def main(train_dir, model_save_dir, backbone_dir, classifier):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    if classifier:
        print("Training classification model...")
        train_classifier(train_dir, model_save_dir)
    else:
        print("Training detection model...")
        train_detector(train_dir, model_save_dir, backbone_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    parser.add_argument("--backbone_dir", type=str)
    parser.add_argument("-c", "--classifier", action="store_true", default=False)
    args = parser.parse_args()
    main(args.train_dir, args.model_save_dir, args.backbone_dir, args.classifier)
