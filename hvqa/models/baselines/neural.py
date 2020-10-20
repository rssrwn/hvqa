import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import ImageFilter

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence

import hvqa.util.func as util
from hvqa.util.exceptions import UnknownQuestionTypeException
from hvqa.models.baselines.interfaces import _AbsBaselineModel
from hvqa.detection.detector import NeuralDetector
from hvqa.models.baselines.datasets import (
    E2EDataset,
    E2EObjDataset,
    TvqaDataset,
    BasicDataset
)
from hvqa.models.baselines.networks import (
    LangLstmNetwork,
    CnnMlpNetwork,
    CnnLstmNetwork,
    Cnn3DMlpNetwork,
    CnnMlpObjNetwork,
    TvqaNetwork,
    MacNetwork
)


class _AbsNeuralModel(_AbsBaselineModel):
    def __init__(self, spec, model):
        super(_AbsNeuralModel, self).__init__(spec)

        gpus = torch.cuda.device_count()
        model = nn.DataParallel(model) if gpus > 1 else model
        print(f"Neural VideoQA model using {gpus} cuda devices.")

        self._device = util.get_device()
        self._model = model.to(self._device)
        self._model.eval()
        self._loss_fn = NLLLoss(reduction="none")
        self._epochs, self._lr, self._batch_size = self._set_hyperparams()
        self._print_freq = 10

    def train(self, train_data, eval_data, verbose=True, save_path=None):
        """
        Train the model

        :param train_data: Training dataset: BaselineDataset
        :param eval_data: Validation dataset: BaselineDataset
        :param verbose: Additional printing while training: bool
        :param save_path: If provided, model will save at the end of each epoch
        """

        print("\nPreparing data...")
        train_loader = self._prepare_train_data(train_data)
        eval_loader = self._prepare_eval_data(eval_data)
        print("Data preparation complete.")

        print("Training Neural baseline VideoQA model...")

        optimiser = optim.Adam(self._model.parameters(), lr=self._lr, eps=1e-5)
        for e in range(self._epochs):
            self._train_one_epoch(train_loader, optimiser, e, verbose)
            print()
            self._eval(eval_loader, verbose)
            if save_path is not None:
                self.save(save_path)

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
                pred = output[q_type][idx].to("cpu").float()
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
            pred = output[q_type][idx][None, :].to("cpu").float()
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
        batch_size = 256
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

        self._model = self._model.half().to(self._device)

        self.video_lstm = video_lstm

        self.transform = T.Compose([
            T.ToTensor()
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
        frames = torch.cat(frames, dim=0).half().to(self._device)
        qs = pack_sequence(questions, enforce_sorted=False).half().to(self._device)
        return frames, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.0001
        batch_size = 16
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


class Cnn3DMlpModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(Cnn3DMlpModel, self).__init__(spec, model)

        self._model = self._model.half().to(self._device)

        self._print_freq = 50
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EDataset.from_baseline_dataset(self.spec, train_data, self.transform, lang_only=False)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EDataset.from_baseline_dataset(self.spec, eval_data, self.transform, lang_only=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        frames = [torch.stack(v_frames).transpose(0, 1) for v_frames in frames]
        frames = torch.stack(frames).half().to(self._device)
        qs = pack_sequence(questions, enforce_sorted=False).half().to(self._device)
        return frames, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.0001
        batch_size = 16
        return epochs, lr, batch_size

    @staticmethod
    def new(spec):
        network = Cnn3DMlpNetwork(spec)
        model = Cnn3DMlpModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(Cnn3DMlpNetwork, model_path, spec)
        model = Cnn3DMlpModel(spec, network)
        return model


class MacModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(MacModel, self).__init__(spec, model)

        self._model = self._model.half().to(self._device)

        self._print_freq = 20
        self.transform = T.Compose([
            T.ToTensor(),
        ])

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EDataset.from_baseline_dataset(self.spec, train_data, self.transform, lang_only=False)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EDataset.from_baseline_dataset(self.spec, eval_data, self.transform, lang_only=False)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=False, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        frames = [torch.stack(v_frames) for v_frames in frames]
        frames = torch.cat(frames, dim=0).half().to(self._device)
        qs = pad_sequence(questions).half().to(self._device)
        return frames, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.0001
        batch_size = 24
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, p=8):
        network = MacNetwork(spec, p)
        model = MacModel(spec, network)
        return model

    @staticmethod
    def load(spec, path, p=8):
        model_path = Path(path) / "network.pt"
        network = util.load_model(MacNetwork, model_path, spec, p)
        model = MacModel(spec, network)
        return model

    # def train(self, train_data, eval_data, verbose=True, save_path=None):
    #     train_loader = self._prepare_train_data(train_data)
    #     dataset = train_loader.dataset


# ********************************************************************************************
# ************************************ Object Models *****************************************
# ********************************************************************************************


class CnnObjModel(_AbsNeuralModel):
    def __init__(self, spec, model, parse_q=False):
        super(CnnObjModel, self).__init__(spec, model)

        self._print_freq = 10
        self.parse_q = parse_q

        self._model = self._model.half().to(self._device)

        num_workers = os.cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EObjDataset.from_video_dataset(self.spec, train_data, parse_q=self.parse_q)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EObjDataset.from_video_dataset(self.spec, eval_data, parse_q=self.parse_q)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        if self.parse_q:
            qs = torch.stack(questions).half().to(self._device)
        else:
            qs = pack_sequence(questions, enforce_sorted=False).half().to(self._device)

        frames = [frame for frames_ in frames for frame in frames_]
        obj_frames = util.gen_object_frames(frames, self._executor)
        obj_frames = torch.stack(obj_frames).to(self._device)
        return obj_frames, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.0001
        batch_size = 128
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, parse_q=False):
        network = CnnMlpObjNetwork(spec, parse_q=parse_q)
        model = CnnObjModel(spec, network, parse_q=parse_q)
        return model

    @staticmethod
    def load(spec, path, parse_q=False):
        model_path = Path(path) / "network.pt"
        network = util.load_model(CnnMlpObjNetwork, model_path, spec, parse_q)
        model = CnnObjModel(spec, network, parse_q=parse_q)
        return model


class TvqaModel(_AbsNeuralModel):
    def __init__(self, spec, model, curr_learning=False):
        super(TvqaModel, self).__init__(spec, model)

        self._model = self._model.half().to(self._device)

        self.curr_learning = curr_learning
        self._print_freq = 10
        self._trans = T.ToTensor()
        self._optim = optim.Adam(self._model.parameters(), lr=self._lr)

    def train(self, train_data, eval_data, verbose=True, save_path=None):
        print("\nPreparing data...")
        train_dataset = TvqaDataset.from_video_dataset(self.spec, train_data)
        eval_dataset = TvqaDataset.from_video_dataset(self.spec, eval_data)
        print("Data preparation complete.")

        if self.curr_learning:
            epochs_q_t = 5
            for q_type in ["property", "relation", "action"]:
                self._train_q_type(train_dataset, eval_dataset, q_type, epochs_q_t)

        print("Training TVQA model...")
        self._print_freq = 50

        fn = util.collate_func
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)

        for e in range(self._epochs):
            self._train_one_epoch(train_loader, self._optim, e, verbose)
            print()
            self._eval(eval_loader, verbose)
            if save_path is not None:
                self.save(save_path)

        print("Training complete.")

    def _train_q_type(self, train_dataset, eval_dataset, q_type, epochs, verbose=True):
        print(f"Training on {q_type} questions...")
        fn = util.collate_func
        train_dataset_ = self._filter_q_type(train_dataset, q_type)
        eval_dataset_ = self._filter_q_type(eval_dataset, q_type)
        train_loader = DataLoader(train_dataset_, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        eval_loader = DataLoader(eval_dataset_, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        for e in range(epochs):
            self._train_one_epoch(train_loader, self._optim, e, verbose)
            print()
            self._eval(eval_loader, verbose)

        print(f"Completed training {q_type} questions.")

    def _filter_q_type(self, tvqa_dataset, q_type):
        if q_type == "property":
            q_type = 0
        elif q_type == "relation":
            q_type = 1
        elif q_type == "action":
            q_type = 2
        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        new_data = []
        for v_idx in range(len(tvqa_dataset)):
            (frames, raw_video), question, q_type_, answer = tvqa_dataset[v_idx]
            if q_type == q_type_:
                new_data.append(((frames, raw_video), question, q_type_, answer))

        dataset = BasicDataset(new_data)
        return dataset

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = TvqaDataset.from_video_dataset(self.spec, train_data)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = TvqaDataset.from_video_dataset(self.spec, eval_data)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        videos, raw_videos = tuple(zip(*frames))

        qs = torch.stack(questions).half().to(self._device)

        v_frames = [frame for video in videos for frame in video]
        v_frames = [[obj for obj, _ in frame] for frame in v_frames]
        v_frames = [torch.stack(frame) for frame in v_frames]
        v_frames = pad_sequence(v_frames).half().to(self._device)

        raw_pairs = [list(zip(video, video[1:])) for video in raw_videos]
        raw_pairs = [[(self._trans(i1), self._trans(i2)) for (i1, i2) in video] for video in raw_pairs]
        raw_pairs = [[torch.cat(pair, dim=0) for pair in video] for video in raw_pairs]
        raw_pairs = [frame_pair for video in raw_pairs for frame_pair in video]
        raw_pairs = torch.stack(raw_pairs).half().to(self._device)

        return v_frames, raw_pairs, qs

    def _set_hyperparams(self):
        devices = torch.cuda.device_count()
        devices = 1 if devices == 0 else devices
        epochs = 10
        lr = 0.0001
        batch_size = 16 * devices
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, curr_learning=False):
        network = TvqaNetwork(spec)
        model = TvqaModel(spec, network, curr_learning=curr_learning)
        return model

    @staticmethod
    def load(spec, path, curr_learing=False):
        model_path = Path(path) / "network.pt"
        network = util.load_model(TvqaNetwork, model_path, spec)
        model = TvqaModel(spec, network, curr_learning=curr_learing)
        return model
