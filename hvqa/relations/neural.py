import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util.interfaces import Component, Trainable
from hvqa.relations.dataset import QARelationDataset
from hvqa.relations.models import RelationClassifierModel
from hvqa.util.func import collate_func, get_device, load_model, save_model, obj_encoding


class NeuralRelationClassifier(Component, Trainable):
    def __init__(self, spec, model):
        super(NeuralRelationClassifier, self).__init__()

        self.spec = spec

        self._loss_fn = nn.MSELoss()
        self._device = get_device()
        self.model = model.to(self._device)

        self._prob_threshold = 0.5

    def run_(self, video):
        for frame in video.frames:
            self._run_frame(frame)

    def _run_frame(self, frame):
        objs_encs = []
        obj_idxs = []
        for idx1, obj1 in enumerate(frame.objs):
            for idx2, obj2 in enumerate(frame.objs):
                if idx1 != idx2:
                    obj_enc = obj_encoding(self.spec, obj1)
                    obj2_enc = obj_encoding(self.spec, obj2)
                    obj_enc.extend(obj2_enc)
                    objs = torch.tensor(obj_enc, dtype=torch.float32)
                    objs_encs.append(objs)
                    obj_idxs.append((idx1, idx2))

        objs_encs = torch.stack(objs_encs).to(self._device)
        with torch.no_grad():
            output = self.model(objs_encs)

        for rel, probs in output.items():
            out_bool = list(torch.BoolTensor(probs.cpu() > self._prob_threshold))
            idxs = [obj_idxs[idx] for idx, out in enumerate(out_bool) if out]
            [frame.set_relation(idx1, idx2, rel) for idx1, idx2 in idxs]

    @staticmethod
    def load(spec, path):
        model = load_model(RelationClassifierModel, path, spec)
        model.eval()
        relations = NeuralRelationClassifier(spec, model)
        return relations

    @staticmethod
    def new(spec, **kwargs):
        model = RelationClassifierModel(spec)
        model.eval()
        relations = NeuralRelationClassifier(spec, model)
        return relations

    def save(self, path):
        save_model(self.model, path)

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

        videos, answers = train_data

        print("\nConstructing training and evaluating relation datasets...")
        qa_loader = self._construct_eval_datasets(eval_data, batch_size)
        train_dataset = QARelationDataset(self.spec, videos, answers)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)

        print(f"Training neural relation classifier with device {self._device}...")

        optimiser = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)
            self._eval(qa_loader)

        print("Completed relation classifier training.")

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose, print_freq=20):
        num_batches = len(train_loader)
        for t, data in enumerate(train_loader):
            self.model.train()

            outputs = {}
            targets = {}
            for rel, rel_data in data.items():
                objs_enc, rel_cls = rel_data
                objs_enc = torch.stack(objs_enc).to(self._device)
                rel_cls = torch.stack(rel_cls).to(self._device)
                output = self.model(objs_enc)
                outputs[rel] = output[rel].to("cpu")
                targets[rel] = rel_cls.to("cpu")

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

    def _construct_eval_datasets(self, eval_data, batch_size):
        qa_data = QARelationDataset.from_video_dataset(self.spec, eval_data, sample=False)
        qa_loader = DataLoader(qa_data, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        return qa_loader

    def eval(self, eval_data, batch_size=256):
        print("\nConstructing evaluation data...")
        qa_loader = self._construct_eval_datasets(eval_data, batch_size)
        self._eval(qa_loader)

    def _eval(self, qa_loader):
        print("\nEvaluating neural relation classifier on QA relation data...")
        self._eval_from_data(qa_loader)

    def _eval_from_data(self, eval_loader):
        self.model.eval()

        rel_outputs = {rel: [] for rel in self.spec.relations}
        rel_targets = {rel: [] for rel in self.spec.relations}
        for t, data in enumerate(eval_loader):
            for rel, (objs_enc, rel_cls) in data.items():
                objs_enc = torch.stack(objs_enc).to(self._device)
                rel_cls = torch.stack(rel_cls).to(self._device)
                with torch.no_grad():
                    output = self.model(objs_enc)

                out = list(output[rel].to("cpu").numpy())
                target = list(rel_cls.to("cpu").numpy())
                rel_outputs[rel].extend(out)
                rel_targets[rel].extend(target)

        rel_metrics = {}
        for rel, outputs in rel_outputs.items():
            targets = rel_targets[rel]
            metrics = self._eval_rel(np.array(outputs), np.array(targets))
            rel_metrics[rel] = metrics

        total = 0
        correct = 0
        for rel, (tp, tn, fp, fn) in rel_metrics.items():
            correct_rel = tp + tn
            total_rel = tp + tn + fp + fn
            correct += correct_rel
            total += total_rel

            tp_perc = f"{tp / total_rel:.2f}"
            tn_perc = f"{tn / total_rel:.2f}"
            fp_perc = f"{fp / total_rel:.2f}"
            fn_perc = f"{fn / total_rel:.2f}"

            print(f"\nConfusion matrix for {rel}:")
            print(f"{'Actual:':<10}")
            print(f"{'Pred:':>10}{'Yes':^10}{'No':^10}")
            print(f"{'Yes':<10}{'TP ' + tp_perc:^10}{'FN ' + fn_perc:^10}")
            print(f"{'No':<10}{'FP ' + fp_perc:^10}{'TN ' + tn_perc:^10}")

            acc = correct_rel / total_rel
            precision = tp / (tp + fp) if (tp + fp) != 0 else "NaN"
            recall = tp / (tp + fn) if (tp + fn) != 0 else "NaN"
            f1 = (2 * precision * recall) / (precision + recall) if precision != "NaN" and recall != "NaN" else "NaN"

            precision = f"{precision:.2f}" if type(precision) == float else precision
            recall = f"{recall:.2f}" if type(recall) == float else recall
            f1 = f"{f1:.2f}" if type(f1) == float else f1

            print(f"\nResults for {rel}:")
            print(f"{'Accuracy:':<10} {acc:.2f}")
            print(f"{'Precision:':<10} {precision}")
            print(f"{'Recall:':<10} {recall}")
            print(f"{'F1 Score:':<10} {f1}")

        overall_acc = correct / total
        print(f"\nOverall accuracy: {overall_acc:.2f}\n")

    def _eval_rel(self, outputs, targets):
        outputs_bool = torch.BoolTensor(outputs > self._prob_threshold)
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
