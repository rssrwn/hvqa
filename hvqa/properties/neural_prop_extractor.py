import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, spec, model, hardcoded=True, lr=0.001, batch_size=128, epochs=1, print_freq=10):
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
                for prop, val in prop_vals.items():
                    obj.set_prop_val(prop, val)

    def _extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of dicts [{prop: val}]
        """

        device = get_device()
        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)
        obj_imgs_batch = obj_imgs_batch.to(device)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        batch_sizes = set([len(out) for _, out in model_out.items()])
        assert len(batch_sizes) == 1, "Number of model predictions must be the same for each property"

        objs = []
        for idx in range(list(batch_sizes)[0]):
            obj = {}
            for prop, pred in model_out.items():
                pred = torch.max(pred, dim=1)[1].cpu().numpy()
                val = self.spec.from_internal(prop, pred[idx])
                obj[prop] = val

            objs.append(obj)

        return objs

    def train(self, train_data, eval_data, verbose=True):
        if self.hardcoded:
            train_dataset = PropDataset(self.spec, train_data, True, transform=_transform)
        else:
            raise NotImplementedError()

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_func)
        optimiser = optim.Adam(self.model.parameters(), lr=self.lr)

        print(f"Training property extraction model using device {self.device}...")

        for epoch in range(self.epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)

            # Save a temp model every epoch
            current_save = f"{self._temp_save}/after_{epoch + 1}_epochs.pt"
            torch.save(self.model.state_dict(), current_save)

            # Evaluate performance every epoch
            self.eval(eval_data)

        print(f"Completed training, final model saved to {current_save}")

    def eval(self, eval_data, threshold=0.5):
        print("Evaluating neural property extraction component...")

        eval_dataset = PropDataset(self.spec, eval_data, True, transform=_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_func)

        self.model.eval()

        # Setup metrics
        num_predictions = 0
        props = self.spec.prop_names()
        losses = {prop: [] for prop in props}
        tps = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        fps = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        tns = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        fns = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        correct = {prop: 0 for prop in props}

        for i, (imgs, objs) in enumerate(eval_loader):
            images, targets = self._prepare_data(imgs, objs)

            with torch.no_grad():
                preds = self.model(images)

            output = {prop: out.to("cpu") for prop, out in preds.items()}
            targets = {prop: target.to("cpu") for prop, target in targets.items()}
            results = {prop: self._eval_prop(pred, targets[prop], threshold) for prop, pred in output.items()}

            for prop, (loss_, tps_, fps_, tns_, fns_, num_correct) in results.items():
                for idx, target in enumerate(self.spec.prop_values(prop)):
                    losses[prop].extend(loss_)
                    tps[prop][target] += tps_[idx]
                    fps[prop][target] += fps_[idx]
                    tns[prop][target] += tns_[idx]
                    fns[prop][target] += fns_[idx]

                correct[prop] += num_correct
            num_predictions += len(imgs)

        results = (tps, fps, tns, fns)
        self._print_results(results, correct, losses, num_predictions)

    @staticmethod
    def new(spec, **kwargs):
        model = PropertyExtractionModel(spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    @staticmethod
    def load(spec, path):
        model = load_model(PropertyExtractionModel, path, spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    def save(self, path):
        save_model(self.model, path)

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        num_batches = len(train_loader)
        for t, (imgs, objs) in enumerate(train_loader):
            self.model.train()

            images, targets = self._prepare_data(imgs, objs)
            output = self.model(images)
            output = {prop: out.to("cpu") for prop, out in output.items()}
            targets = {prop: target.to("cpu") for prop, target in targets.items()}

            loss, losses = self._calc_loss(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % self.print_freq == 0:
                loss_str = f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.6f}"
                for prop, loss in losses.items():
                    loss_str += f" -- {prop} loss = {loss.item():.6f}"
                print(loss_str)

    def _prepare_data(self, imgs, objs):
        targets = {}

        # Convert to tensors
        images = torch.cat([img[None, :, :, :] for img in imgs])
        for prop in self.spec.prop_names():
            prop_vals = [torch.tensor([self.spec.to_internal(prop, obj[prop])]) for obj in objs]
            targets[prop] = torch.cat(prop_vals)

        # Send to device
        images = images.to(device=self.device)
        targets = {prop: vals.to(self.device) for prop, vals in targets.items()}

        return images, targets

    def _calc_loss(self, preds, targets):
        losses = {prop: self.loss_fn(pred, targets[prop]) for prop, pred in preds.items()}
        loss = sum(losses.values())
        return loss, losses

    @staticmethod
    def _eval_prop(preds, indices, threshold=None):
        """
        Calculate metrics for classification of a single property

        :param preds: Network predictions (tensor of floats, (N, C))
        :param indices: Target class indices (tensor of ints, (N))
        :return: loss (list of floats), TP, FP, TN, FN (all 1d tensors of ints), num_correct (int)
        """

        if threshold is None:
            threshold = 0

        preds_shape = preds.shape
        targets_shape = indices.shape

        assert preds_shape[0] == targets_shape[0], "Predictions and targets must have the same batch size"

        loss = F.cross_entropy(preds, indices, reduction="none")
        loss_batch = list(loss.numpy())

        preds = F.softmax(preds, dim=1)

        # Convert targets to one-hot encoding
        targets = torch.eye(preds_shape[1]).index_select(0, indices)

        act_bool = torch.BoolTensor(targets == 1)
        max_vals, _ = torch.max(preds, 1)
        preds_bool = torch.BoolTensor(preds >= max_vals[:, None])
        preds_bool = preds_bool & (preds >= threshold)

        _, pred_idxs = torch.max(preds_bool, dim=1)
        _, act_idxs = torch.max(act_bool, dim=1)

        num_correct = torch.sum(act_idxs == pred_idxs)
        tps = torch.sum(act_bool & preds_bool, dim=0)
        fps = torch.sum(~act_bool & preds_bool, dim=0)
        tns = torch.sum(~act_bool & ~preds_bool, dim=0)
        fns = torch.sum(act_bool & ~preds_bool, dim=0)

        return loss_batch, tps, fps, tns, fns, num_correct.item()

    def _print_results(self, results, correct, losses, num_predictions):
        tps, fps, tns, fns = results
        for prop in self.spec.prop_names():
            print(f"\nResults for {prop}:")

            loss = losses[prop]
            class_tps = tps[prop]
            class_fps = fps[prop]
            class_tns = tns[prop]
            class_fns = fns[prop]

            print(f"{'Value':<12}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}")

            for val in self.spec.prop_values(prop):
                tp = class_tps[val].item()
                fp = class_fps[val].item()
                tn = class_tns[val].item()
                fn = class_fns[val].item()

                precision = tp / (tp + fp) if (tp + fp) != 0 else float("NaN")
                recall = tp / (tp + fn) if (tp + fn) != 0 else float("NaN")
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                print(f"{val:<12}{accuracy:<12.4f}{precision:<12.4f}{recall:<12.4f}")

            avg_loss = torch.tensor(loss).mean()
            acc = correct[prop] / num_predictions

            print(f"\nAverage loss: {avg_loss:.6f}")
            print(f"Overall accuracy: {acc:.4f}\n")
