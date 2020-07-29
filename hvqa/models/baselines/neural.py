from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

import hvqa.util.func as util
from hvqa.models.baselines.datasets import EndToEndDataset
from hvqa.models.baselines.networks import LangLstmNetwork, CnnMlpNetwork
from hvqa.models.baselines.interfaces import _AbsBaselineModel


class _AbsNeuralModel(_AbsBaselineModel):
    def __init__(self, spec, model):
        super(_AbsNeuralModel, self).__init__(spec)
        self._device = util.get_device()
        self._model = model.to(self._device)
        self._model.eval()
        self._loss_fn = NLLLoss(reduction="none")
        self._epochs, self._lr = self._set_hyperparams()

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the LSTM model

        :param train_data: Training dataset: BaselineDataset
        :param eval_data: Validation dataset: BaselineDataset
        :param verbose: Additional printing while training: bool
        """

        print("Preparing data...")
        train_loader = self._prepare_train_data(train_data)
        eval_loader = self._prepare_eval_data(eval_data)
        print("Data preparation complete.")

        print("Training Language LSTM model...")

        optimiser = optim.Adam(self._model.parameters(), lr=self._lr)
        for e in range(self._epochs):
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

    def _set_hyperparams(self):
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


class LangLstmModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(LangLstmModel, self).__init__(spec, model)

    def _prepare_train_data(self, train_data, batch_size=64):
        train_dataset = EndToEndDataset.from_baseline_dataset(self.spec, train_data, lang_only=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=util.collate_func)
        return train_loader

    def _prepare_eval_data(self, eval_data, batch_size=64):
        eval_dataset = EndToEndDataset.from_baseline_dataset(self.spec, eval_data, lang_only=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=util.collate_func)
        return eval_loader

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose, print_freq=10):
        self._model.train()
        num_batches = len(train_loader)
        for t, (qs, q_types, ans) in enumerate(train_loader):
            qs = pack_sequence(qs, enforce_sorted=False).to(self._device)
            output = self._model(qs)
            loss = self._calc_loss(output, q_types, ans)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % print_freq == 0:
                print(f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.4f}")

    def _eval(self, eval_loader, verbose):
        self._model.eval()
        num_batches = len(eval_loader)

        for t, (qs, q_types, ans) in enumerate(eval_loader):
            qs = pack_sequence(qs, enforce_sorted=False).to(self._device)

            with torch.no_grad():
                output = self._model(qs)

            results = []
            for idx, q_type in enumerate(q_types):
                answer = ans[idx].item()
                pred = output[q_type][idx].to("cpu")
                _, max_idx = torch.max(pred, 0)
                results.append(("", q_type, max_idx.item(), answer))

            self._eval_video_results(t, num_batches, results, False)
        self._print_results()

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        return epochs, lr

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


class CnnMlpModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(CnnMlpModel, self).__init__(spec, model)

        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def _prepare_train_data(self, train_data, batch_size=8):
        train_dataset = EndToEndDataset.from_baseline_dataset(self.spec, train_data,
                                                              lang_only=False, transform=self.transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=util.collate_func)
        return train_loader

    def _prepare_eval_data(self, eval_data, batch_size=8):
        eval_dataset = EndToEndDataset.from_baseline_dataset(self.spec, eval_data,
                                                             lang_only=False, transform=self.transform)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose, print_freq=50):
        self._model.train()
        num_batches = len(train_loader)
        for t, (frames, qs, q_types, ans) in enumerate(train_loader):
            frames = [torch.stack(v_frames) for v_frames in frames]
            frames = torch.cat(frames, dim=0).to(self._device)
            qs = pack_sequence(qs, enforce_sorted=False).to(self._device)

            output = self._model((frames, qs))
            loss = self._calc_loss(output, q_types, ans)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % print_freq == 0:
                print(f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.4f}")

    def _eval(self, eval_loader, verbose):
        self._model.eval()
        num_batches = len(eval_loader)

        for t, (frames, qs, q_types, ans) in enumerate(eval_loader):
            frames = [torch.stack(v_frames) for v_frames in frames]
            frames = torch.cat(frames, dim=0).to(self._device)
            qs = pack_sequence(qs, enforce_sorted=False).to(self._device)

            with torch.no_grad():
                output = self._model((frames, qs))

            results = []
            for idx, q_type in enumerate(q_types):
                answer = ans[idx].item()
                pred = output[q_type][idx].to("cpu")
                _, max_idx = torch.max(pred, 0)
                results.append(("", q_type, max_idx.item(), answer))

            self._eval_video_results(t, num_batches, results, False)
        self._print_results()

    def _set_hyperparams(self):
        epochs = 5
        lr = 0.001
        return epochs, lr

    @staticmethod
    def new(spec):
        network = CnnMlpNetwork(spec)
        model = CnnMlpModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(CnnMlpNetwork, model_path, spec)
        model = CnnMlpModel(spec, network)
        return model

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True)
        model_path = path / "network.pt"
        util.save_model(self._model, model_path)
