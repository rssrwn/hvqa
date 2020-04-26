import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from pathlib import Path

from hvqa.properties.dataset import PropDataset
from hvqa.properties.models import PropertyExtractionModel
from hvqa.util.func import get_device, load_model, save_model, collate_func
from hvqa.util.interfaces import Component


_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


class NeuralPropExtractor(Component):
    def __init__(self, spec, model, hardcoded=True, lr=0.001, batch_size=128, epochs=5, print_freq=10):
        super(NeuralPropExtractor, self).__init__()

        self.device = get_device()

        self.spec = spec
        self.model = model.to(self.device)
        self.hardcoded = hardcoded
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.print_freq = print_freq

        self.loss_fn = nn.CrossEntropyLoss()

        self._temp_save = Path("saved-models/properties/temp")
        self._temp_save.mkdir(exist_ok=True, parents=True)

    def run_(self, video):
        for frame in video.frames:
            objs = frame.objs
            obj_imgs = [obj.img for obj in objs]
            props = self._extract_props(obj_imgs)
            for idx, prop_vals in enumerate(props):
                obj = objs[idx]
                for prop_idx, val in enumerate(prop_vals):
                    prop_name = self.spec.prop_names[prop_idx]
                    obj.set_prop_val(prop_name, val)

    def _extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of tuple [(prop_val1, prop_val2, ...)]
        """

        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)

        device = get_device()
        obj_imgs_batch = obj_imgs_batch.to(device)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        preds = [torch.max(pred, dim=1)[1].cpu().numpy() for pred in model_out]

        prop_list = []
        length = None
        for prop_idx, pred in enumerate(preds):
            vals = [self.spec.prop_values[idx] for idx in pred]
            length = len(vals) if length is None else length
            assert length == len(vals), "Number of predictions must be the same"
            prop_list.append(vals)

        props = zip(prop_list)
        return props

    def train(self, train_data, eval_data, verbose=True):
        if self.hardcoded:
            train_dataset = PropDataset(self.spec, train_data, True, transform=_transform)
            eval_dataset = PropDataset(self.spec, eval_data, True, transform=_transform)
        else:
            raise NotImplementedError()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_func)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_func)
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"Training property extraction model using device {self.device}...")

        for epoch in range(self.epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)

            # Save a temp model every epoch
            current_save = f"{self._temp_save}/after_{epoch + 1}_epochs.pt"
            torch.save(self.model.state_dict(), current_save)

            # Evaluate performance every epoch
            # evaluator.eval_model(model)

        print(f"Completed training, final model saved to {current_save}")

    @staticmethod
    def new(spec, **kwargs):
        model = PropertyExtractionModel(spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    @staticmethod
    def load(path, spec):
        model = load_model(PropertyExtractionModel, path, (spec,))
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    def save(self, path):
        save_model(self.model, path)

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        num_batches = len(train_loader)
        for t, (x, y) in enumerate(train_loader):
            self.model.train()

            images = torch.cat([img[None, :, :, :] for img in x])

            targets = []
            for obj in y:
                obj = {prop: self.spec.to_internal(prop, val) for prop, val in obj.items()}
                obj = {prop: torch.tensor([val]) for prop, val in obj.items()}
                targets.append(obj)

            targets = {prop: torch.cat([obj[prop] for obj in targets]) for prop in self.spec.prop_names()}
            images = images.to(device=self.device)
            targets = {prop: vals.to(self.device) for prop, vals in targets.items()}

            output = self.model(images)
            output = {self.spec.prop_names()[idx]: out.to("cpu") for idx, out in enumerate(output)}

            loss, losses = self._calc_loss(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % self.print_freq == 0:
                loss_str = f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.6f}"
                for prop, loss in losses.items():
                    loss_str += f" -- {prop} loss = {loss.item():.6f}"
                print(loss_str)

    def _calc_loss(self, preds, targets):
        losses = {prop: self.loss_fn(pred, targets[prop]) for prop, pred in preds.items()}
        loss = sum(losses.values())
        return loss, losses
