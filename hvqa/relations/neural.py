import torch.optim as optim
from torch.utils.data import DataLoader

from hvqa.util.interfaces import Component, Trainable
from hvqa.relations.dataset import RelationDataset
from hvqa.relations.models import RelationClassifierModel
from hvqa.util.func import collate_func


class NeuralRelationClassifier(Component, Trainable):
    def __init__(self, spec):
        super(NeuralRelationClassifier, self).__init__()

        self.spec = spec

        self.model = RelationClassifierModel(spec)

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

        videos, answers = train_data
        train_dataset = RelationDataset(self.spec, videos, answers)
        eval_dataset = RelationDataset.from_video_dataset(self.spec, eval_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=self._collate_fn)
        optimiser = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)
            self.eval(eval_loader)

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        pass

    def eval(self, eval_loader):
        pass

    @staticmethod
    def _collate_fn(data):
        out_data = [[] for _ in range(len(data[0]))]
        for item in data:
            for rel_idx, rel_data in enumerate(item):
                out_data[rel_idx].append(rel_data)

        out_data = [collate_func(batch) for batch in out_data]
        return out_data
