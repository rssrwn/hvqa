from pathlib import Path

import torch
import torch.optim as optim
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

import hvqa.util.func as util
from hvqa.util.exceptions import UnknownQuestionTypeException
from hvqa.models.baselines.datasets import EndToEndDataset
from hvqa.models.baselines.networks import LangLstmNetwork

from hvqa.models.baselines.interfaces import _AbsBaselineModel


class _AbsNeuralModel(_AbsBaselineModel):
    def __init__(self, spec, model):
        super(_AbsNeuralModel, self).__init__(spec)
        self._device = util.get_device()
        self._model = model.to(self._device)
        self._model.eval()
        self._loss_fn = NLLLoss(reduction="none")

    def train(self, train_data, eval_data, verbose=True, epochs=10, lr=0.001):
        """
        Train the LSTM model

        :param train_data: Training dataset: BaselineDataset
        :param eval_data: Validation dataset: BaselineDataset
        :param verbose: Additional printing while training: bool
        :param epochs: Number of training epochs
        :param lr: Learning rate for training
        """

        print("Preparing data...")
        train_loader = self._prepare_train_data(train_data)
        eval_loader = self._prepare_eval_data(eval_data)
        print("Data preparation complete.")

        print("Training Language LSTM model...")

        optimiser = optim.Adam(self._model.parameters(), lr=lr)
        for e in range(epochs):
            self._train_one_epoch(train_loader, optimiser, e, verbose)
            print()
            self._eval(eval_loader, verbose)

        print("Training complete.")

    def eval(self, eval_data, verbose=True):
        eval_loader = self._prepare_eval_data(eval_data)
        self._eval(eval_loader, verbose)

    def _prepare_train_data(self, train_data):
        raise NotImplementedError()

    def _prepare_eval_data(self, eval_data):
        raise NotImplementedError()

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        raise NotImplementedError()

    def _eval(self, eval_loader, verbose):
        raise NotImplementedError()

    @staticmethod
    def new(spec):
        raise NotImplementedError()

    @staticmethod
    def load(spec, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def _calc_loss(self, output, q_types, ans):
        losses = []
        for idx, q_type in enumerate(q_types):
            target = ans[idx][None].to("cpu")
            pred = output[q_type][idx][None, :].to("cpu")
            loss = self._loss_fn(pred, target)
            losses.append(loss)

        batch_size = len(losses)
        loss = sum(losses) / batch_size
        return loss


class CNNMLPModel(_AbsNeuralModel):
    pass


class LangLstmModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(LangLstmModel, self).__init__(spec, model)

    def _prepare_train_data(self, train_data):
        pass

    def _prepare_eval_data(self, eval_data):
        pass

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        pass

    def _eval(self, eval_loader, verbose):
        pass

    @staticmethod
    def new(spec):
        network = LangLstmNetwork(spec)
        model = LangLstmModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(LangLstmNetwork, model_path, spec)
        model = LangLstmModel(spec, network)
        return model

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        model_path = path / "network.pt"
        util.save_model(self._model, model_path)
