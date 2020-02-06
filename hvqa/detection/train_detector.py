import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T

from hvqa.util import get_device, load_model, add_edges, NUM_YOLO_REGIONS
from hvqa.detection.hyperparameters import *
from hvqa.detection.models import DetectionModel, ClassifierModel, DetectionBackbone
from hvqa.detection.dataset import DetectionDataset

from lib.vision.engine import train_one_epoch, evaluate


PRINT_BATCHES = 50

_mse_func = nn.MSELoss(reduction="none")

detector_transforms = T.Compose([
    # T.Lambda(lambda x: add_edges(x)),
    T.ToTensor(),
])


def train_detector(train_loader, test_loader, model, optimiser, model_save, scheduler=None, epochs=100):
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
            loss = sum([loss * REG_MULT if key == "loss_box_reg" else loss for key, loss in loss_dict.items()])

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if t % PRINT_BATCHES == 0:
                losses = [loss * REG_MULT if key == "loss_box_reg" else loss for key, loss in loss_dict.items()]
                losses = torch.tensor(losses)

                print(f"Epoch {e:>3}, batch {t:>4} "
                      f"-- loss {loss:.4f}, "
                      f"loss_classifier {losses[0]:.4f}, loss_box_reg {losses[1]:.4f}, "
                      f"loss_objectness {losses[2]:.4f}, loss_rpn_box_reg {losses[3]:.4f} "
                      f"-- lr {optimiser.param_groups[0]['lr']:.6f}")

        if scheduler is not None:
            scheduler.step()

        # Save a temp model every epoch
        current_save = f"{model_save}/after_{e + 1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

        # Evaluate model after each epoch
        evaluate(model, test_loader, device)

    print(f"Completed training, final model saved to {current_save}")


def collate_func(batch):
    return tuple(zip(*batch))


def main(train_dir,  backbone_dir, model_save_dir):
    # Create model save path, just in case
    path = Path(f"./{model_save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    print("Training detection model...")

    # Read backbone model
    device = get_device()
    pretrained = load_model(ClassifierModel, backbone_dir)
    backbone = DetectionBackbone(pretrained)
    model = DetectionModel(backbone)
    model = model.to(device=device)

    # Read train data
    dataset_train = DetectionDataset(train_dir, NUM_YOLO_REGIONS, transforms=detector_transforms)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    # Read test data
    dataset_test = DetectionDataset(train_dir, NUM_YOLO_REGIONS, transforms=detector_transforms)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_detector(loader_train, loader_test, model, optimiser, model_save_dir)

    # num_epochs = 50
    # for epoch in range(num_epochs):
    #     train_one_epoch(model, optimiser, loader, device, epoch, print_freq=20)
    #     current_save = f"{model_save_dir}/after_{epoch + 1}_epochs.pt"
    #     torch.save(model.state_dict(), current_save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training object detector model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("backbone_dir", type=str)
    parser.add_argument("model_save_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.backbone_dir, args.model_save_dir)
