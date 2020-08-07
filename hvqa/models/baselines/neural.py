import PIL.Image as Im
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

import hvqa.util.func as util
from hvqa.models.baselines.datasets import E2EDataset, E2EFilterDataset
from hvqa.models.baselines.interfaces import _AbsBaselineModel
from hvqa.models.baselines.networks import (
    LangLstmNetwork,
    CnnMlpNetwork,
    CnnLstmNetwork,
    PropRelNetwork,
    EventNetwork,
    PreTrainCnnMlpNetwork
)


class _AbsNeuralModel(_AbsBaselineModel):
    def __init__(self, spec, model):
        super(_AbsNeuralModel, self).__init__(spec)
        self._device = util.get_device()
        self._model = model.to(self._device)
        self._model.eval()
        self._loss_fn = NLLLoss(reduction="none")
        self._epochs, self._lr, self._batch_size = self._set_hyperparams()
        self._print_freq = 10

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

        print("Training Neural baseline VideoQA model...")

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

    def _prepare_input(self, frames, questions, q_types, answers):
        raise NotImplementedError()

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose):
        self._model.train()
        num_batches = len(train_loader)
        for t, (frames, qs, q_types, ans) in enumerate(train_loader):
            model_input = self._prepare_input(frames, qs, q_types, ans)

            output = self._model(model_input)
            loss = self._calc_loss(output, q_types, ans)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % self._print_freq == 0:
                print(f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.4f}")

    def _eval(self, eval_loader, verbose):
        self._model.eval()
        num_batches = len(eval_loader)

        for t, (frames, qs, q_types, ans) in enumerate(eval_loader):
            model_input = self._prepare_input(frames, qs, q_types, ans)

            with torch.no_grad():
                output = self._model(model_input)

            results = []
            for idx, q_type in enumerate(q_types):
                answer = ans[idx].item()
                pred = output[q_type][idx].to("cpu")
                _, max_idx = torch.max(pred, 0)
                results.append(("", q_type, max_idx.item(), answer))

            self._eval_video_results(t, num_batches, results, False)
        self._print_results()

    def _set_hyperparams(self):
        """
        Return the model's hyperparameters

        :return: epochs, lr, batch size
        """

        raise NotImplementedError()

    @staticmethod
    def new(spec):
        raise NotImplementedError()

    @staticmethod
    def load(spec, path):
        raise NotImplementedError()

    def save(self, path):
        path = Path(path)
        path.mkdir(exist_ok=True, parents=True)
        model_path = path / "network.pt"
        util.save_model(self._model, model_path)

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

    def _prepare_train_data(self, train_data):
        train_dataset = E2EDataset.from_baseline_dataset(self.spec, train_data, lang_only=True)
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EDataset.from_baseline_dataset(self.spec, eval_data, lang_only=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=False, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        qs = pack_sequence(questions, enforce_sorted=False).to(self._device)
        return qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 64
        return epochs, lr, batch_size

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


class CnnMlpModel(_AbsNeuralModel):
    def __init__(self, spec, model, video_lstm=False):
        super(CnnMlpModel, self).__init__(spec, model)

        self.video_lstm = video_lstm
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def _prepare_train_data(self, train_data):
        train_dataset = E2EDataset.from_baseline_dataset(self.spec, train_data, self.transform, lang_only=False)
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EDataset.from_baseline_dataset(self.spec, eval_data, self.transform, lang_only=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        frames = [torch.stack(v_frames) for v_frames in frames]
        frames = torch.cat(frames, dim=0).to(self._device)
        qs = pack_sequence(questions, enforce_sorted=False).to(self._device)
        return frames, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 8
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, video_lstm=False):
        if video_lstm:
            network = CnnLstmNetwork(spec)
        else:
            network = CnnMlpNetwork(spec)

        model = CnnMlpModel(spec, network)
        return model

    @staticmethod
    def load(spec, path, video_lstm=False):
        model_path = Path(path) / "network.pt"
        if video_lstm:
            network = util.load_model(CnnLstmNetwork, model_path, spec)
        else:
            network = util.load_model(CnnMlpNetwork, model_path, spec)

        model = CnnMlpModel(spec, network)
        return model


class PropRelModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(PropRelModel, self).__init__(spec, model)

        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self._print_freq = 1

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EFilterDataset.from_baseline_dataset(self.spec, train_data, self.transform, filter_qs=[0, 1])
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EFilterDataset.from_baseline_dataset(self.spec, eval_data, self.transform, filter_qs=[0, 1])
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        frames = torch.stack(frames).to(self._device)
        qs = pack_sequence(questions, enforce_sorted=False).to(self._device)
        return frames, qs

    def _set_hyperparams(self):
        epochs = 25
        lr = 0.001
        batch_size = 256
        return epochs, lr, batch_size

    @staticmethod
    def new(spec):
        network = PropRelNetwork(spec)
        model = PropRelModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(PropRelNetwork, model_path, spec)
        model = PropRelModel(spec, network)
        return model


class EventModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(EventModel, self).__init__(spec, model)

        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self._print_freq = 1

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EFilterDataset.from_baseline_dataset(self.spec, train_data, self.transform, filter_qs=[2])
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EFilterDataset.from_baseline_dataset(self.spec, eval_data, self.transform, filter_qs=[2])
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        frames = torch.stack(frames).to(self._device)
        return frames

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 64
        return epochs, lr, batch_size

    @staticmethod
    def new(spec):
        network = EventNetwork(spec)
        model = EventModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(EventNetwork, model_path, spec)
        model = EventModel(spec, network)
        return model


class PreTrainCnnMlpModel(_AbsNeuralModel):
    def __init__(self, spec, model, frame_feat_extr, event_feat_extr):
        super(PreTrainCnnMlpModel, self).__init__(spec, model)

        self.frame_feat_extr = frame_feat_extr.to(self._device)
        for param in self.frame_feat_extr.parameters():
            param.requires_grad = False

        self.event_feat_extr = event_feat_extr.to(self._device)
        for param in self.event_feat_extr.parameters():
            param.requires_grad = False

        self.frame_transform = T.Compose([
            T.ToTensor(),
        ])
        self._print_freq = 1

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EDataset.from_baseline_dataset(self.spec, train_data)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EDataset.from_baseline_dataset(self.spec, eval_data)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        v_feats = self._extr_feats(frames)

        frame_pairs = [zip(v_frames, v_frames[1:]) for v_frames in frames]
        frame_pairs = [[Im.blend(im1, im2, 0.75) for im1, im2 in v_frames] for v_frames in frame_pairs]
        pair_feats = self._extr_feats(frame_pairs)

        feats = torch.cat((v_feats, pair_feats), dim=1).to(self._device)
        qs = pack_sequence(questions, enforce_sorted=False).to(self._device)
        return feats, qs

    def _extr_feats(self, frames):
        frames_ = [[self.frame_transform(frame) for frame in v_frames] for v_frames in frames]
        frames_ = [torch.stack(v_frames).to(self._device) for v_frames in frames_]
        frame_feats = [self.feat_extr(v_frames) for v_frames in frames_]
        v_feats = [frame_feat.reshape(-1) for frame_feat in frame_feats]
        v_feats = torch.stack(v_feats)
        return v_feats

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 256
        return epochs, lr, batch_size

    @staticmethod
    def _load_feat_extr(spec):
        prop_rel = util.load_model(PropRelNetwork, "saved-models/pre/prop-rel/network.pt", spec)
        event = util.load_model(EventNetwork, "saved-models/pre/event/network.pt", spec)
        frame_feat_extr = prop_rel.feat_extr
        event_feat_extr = event.feat_extr
        return frame_feat_extr, event_feat_extr

    @staticmethod
    def new(spec):
        network = PreTrainCnnMlpNetwork(spec)
        frame_feat_extr, event_feat_extr = PreTrainCnnMlpModel._load_feat_extr(spec)
        model = PreTrainCnnMlpModel(spec, network, frame_feat_extr, event_feat_extr)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(PreTrainCnnMlpNetwork, model_path, spec)
        frame_feat_extr, event_feat_extr = PreTrainCnnMlpModel._load_feat_extr(spec)
        model = PreTrainCnnMlpModel(spec, network, frame_feat_extr, event_feat_extr)
        return model
