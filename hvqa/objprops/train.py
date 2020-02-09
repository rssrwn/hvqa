import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as T

from hvqa.util import get_device, collate_func
from hvqa.objprops.dataset import PropertyExtractionDataset
from hvqa.objprops.models import PropertyExtractionModel
from hvqa.objprops.evaluation import PropertyExtractionEvaluator


BATCH_SIZE = 128
LEARNING_RATE = 0.001
PRINT_FREQ = 20


transforms = T.Compose([
    T.Resize((16, 16)),
    T.ToTensor(),
])


_ce_loss = nn.CrossEntropyLoss()


def calc_loss(pred, target):
    act_colour, act_rot, act_cls = target
    pred_colour, pred_rot, pred_cls = pred
    loss_colour = _ce_loss(pred_colour, act_colour)
    loss_rot = _ce_loss(pred_rot, act_rot)
    loss_cls = _ce_loss(pred_cls, act_cls)
    return sum([loss_colour, loss_rot, loss_cls])


def train_one_epoch(model, optimiser, loader_train, device, epoch, print_freq=10):
    for t, (x, y) in enumerate(loader_train):
        model.train()

        images = torch.cat([img[None, :, :, :] for img in x])
        targets = [{k: v.to("cpu") for k, v in t.items()} for t in y]

        colour_targets = torch.cat([target["colour"][None, :] for target in targets])
        rot_targets = torch.cat([target["rotation"][None, :] for target in targets])
        class_targets = torch.cat([target["class"][None, :] for target in targets])

        images = images.to(device=device)
        colour_targets = colour_targets.to(device=device)
        rot_targets = rot_targets.to(device=device)
        class_targets = class_targets.to(device=device)

        output = model(images)
        loss = calc_loss(output, (colour_targets[:,0], rot_targets[:,0], class_targets[:,0]))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if t % print_freq == 0:
            print(f"Epoch {epoch:>3}, batch {t:>4} "
                  f"-- loss = {loss.item():.6f} "
                  f"-- lr = {optimiser.param_groups[0]['lr']:.4f}")


def train_extractor(model, loader_train, loader_test, model_save_dir, epochs=10):
    evaluator = PropertyExtractionEvaluator(loader_test)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = get_device()

    print(f"Training property extraction model using device {device}...")

    model = model.to(device=device)
    for epoch in range(epochs):
        model.train()
        train_one_epoch(model, optimiser, loader_train, device, epoch, print_freq=PRINT_FREQ)

        # Save a temp model every epoch
        current_save = f"{model_save_dir}/after_{epoch + 1}_epochs.pt"
        torch.save(model.state_dict(), current_save)

        # Evaluate performance every epoch
        model.eval()
        evaluator.eval_model(model)

    print(f"Completed training, final model saved to {current_save}")


def main(train_dir, test_dir, save_dir):
    # Create model save path, in case it doesn't exist
    path = Path(f"./{save_dir}")
    path.mkdir(parents=True, exist_ok=True)

    model = PropertyExtractionModel()

    # Read train data
    dataset_train = PropertyExtractionDataset(train_dir, transforms=transforms)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    # Read test data
    dataset_test = PropertyExtractionDataset(test_dir, transforms=transforms)
    loader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_func)

    train_extractor(model, loader_train, loader_test, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training the property extraction network")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("save_model_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.test_dir, args.save_model_dir)
