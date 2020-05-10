import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util.interfaces import Component, Trainable
from hvqa.relations.dataset import RelationDataset
from hvqa.relations.models import RelationClassifierModel
from hvqa.util.func import collate_func, get_device


class NeuralRelationClassifier(Component, Trainable):
    def __init__(self, spec):
        super(NeuralRelationClassifier, self).__init__()

        self.spec = spec

        self._loss_fn = nn.MSELoss()
        self._device = get_device()
        self.model = RelationClassifierModel(spec).to(self._device)

    def run_(self, video):
        pass

    @staticmethod
    def load(spec, path):
        pass

    @staticmethod
    def new(spec, **kwargs):
        pass

    def train(self, train_data, eval_data, verbose=True, lr=0.001, batch_size=256, epochs=10):
        """
        Train the relation classification component

        :param train_data: Training dataset ((Videos, Answers))
        :param eval_data: Evaluation dataset (QADataset)
        :param verbose: Print additional info during training
        :param lr: Learning rate for training
        :param batch_size: Batch size
        :param epochs: Number of epochs to train for
        """

        print(f"Training neural relation classifier with device {self._device}...")

        videos, answers = train_data
        train_dataset = RelationDataset(self.spec, videos, answers)
        eval_dataset = RelationDataset.from_video_dataset(self.spec, eval_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        optimiser = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)
            self.eval(eval_loader)

        print("Completed training.")

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose, print_freq=50):
        num_batches = len(train_loader)
        for t, data in enumerate(train_loader):
            self.model.train()

            outputs = {}
            targets = {}
            for rel, (objs_enc, rel_cls) in data.items():
                objs_enc = objs_enc.to(self._device)
                rel_cls = rel_cls.to(self._device)
                output = self.model(objs_enc)
                outputs[rel] = output[rel].to("cpu").numpy()
                targets[rel] = rel_cls.tp("cpu").numpy()

            loss, losses = self._calc_loss(outputs, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % print_freq == 0:
                loss_str = f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.6f}"
                for rel, loss in losses.items():
                    loss_str += f" -- {rel} loss = {loss.item():.6f}"
                print(loss_str)

    def _calc_loss(self, outputs, targets):
        losses = {rel: self._loss_fn(output, targets[rel]) for rel, output in outputs.items()}
        loss = sum(losses.values())
        return loss, losses

    def eval(self, eval_loader):
        self.model.eval()

        print("Evaluating neural relation classifier...")

        rel_outputs = {rel: [] for rel in self.spec.relations}
        rel_targets = {rel: [] for rel in self.spec.relations}
        for t, data in enumerate(eval_loader):
            for rel, (objs_enc, rel_cls) in data.items():
                objs_enc = objs_enc.to(self._device)
                rel_cls = rel_cls.to(self._device)
                with torch.no_grad():
                    output = self.model(objs_enc)

                out = output[rel].to("cpu").numpy()
                target = rel_cls.tp("cpu").numpy()
                rel_outputs[rel].append(out)
                rel_targets[rel].append(target)

        rel_metrics = {}
        for rel, outputs in rel_outputs.items():
            targets = rel_targets[rel]
            metrics = self._eval_rel(outputs, targets)
            rel_metrics[rel] = metrics

        total = 0
        correct = 0
        for rel, (tp, tn, fp, fn) in rel_metrics.items():
            correct_rel = tp + tn
            total_rel = tp + tn + fp + fn
            correct += correct_rel
            total += total_rel

            acc = correct_rel / total_rel
            precision = tp / (tp + fp) if (tp + fp) != 0 else "NaN"
            recall = tp / (tp + fn) if (tp + fn) != 0 else "NaN"
            f1 = (2 * precision * recall) / (precision + recall) if precision != "NaN" and recall != "NaN" else "NaN"

            print(f"\nResults for {rel}")
            print(f"{'Accuracy':<15}: {acc:.2f}")
            print(f"{'Precision':<15}: {precision:.2f}")
            print(f"{'Recall':<15}: {recall:.2f}")
            print(f"{'F1 Score':<15}: {f1:.2f}")

        overall_acc = correct / total
        print(f"\nOverall accuracy: {overall_acc:.2f}")

    def _eval_rel(self, outputs, targets):
        outputs_bool = torch.BoolTensor(outputs >= 0.5)
        targets_bool = torch.BoolTensor(targets == 1.0)
        num_correct = torch.sum(outputs_bool == targets_bool).item()
        tp = torch.sum(outputs_bool & targets_bool).item()
        tn = torch.sum(~outputs_bool & ~targets_bool).item()
        fp = torch.sum(outputs_bool & ~targets_bool).item()
        fn = torch.sum(~outputs_bool & targets_bool).item()

        assert num_correct == tp + tn

        return tp, tn, fp, fn

    def _collate_fn(self, data):
        out_data = {rel: [] for rel in self.spec.relations}
        for item in data:
            for rel, rel_data in item.items():
                out_data[rel].append(rel_data)

        out_data = {rel: collate_func(batch) for rel, batch in out_data.items()}
        return out_data
