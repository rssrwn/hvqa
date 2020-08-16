import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.nn import NLLLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence, pad_sequence, pad_packed_sequence

import hvqa.util.func as util
from hvqa.models.baselines.interfaces import _AbsBaselineModel
from hvqa.models.baselines.datasets import (
    E2EDataset,
    E2EFilterDataset,
    E2EPreDataset,
    E2EObjDataset,
    E2EObjFilterDataset
)
from hvqa.models.baselines.networks import (
    LangLstmNetwork,
    CnnMlpNetwork,
    CnnLstmNetwork,
    PropRelNetwork,
    EventNetwork,
    CnnMlpPreNetwork,
    CnnMlpObjNetwork,
    PropRelObjNetwork
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

    def train(self, train_data, eval_data, verbose=True, save_path=None):
        """
        Train the LSTM model

        :param train_data: Training dataset: BaselineDataset
        :param eval_data: Validation dataset: BaselineDataset
        :param verbose: Additional printing while training: bool
        :param save_path: If provided, model will save at the end of each epoch
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


class CnnMlpPreModel(_AbsNeuralModel):
    def __init__(self, spec, model, parse_q=False):
        super(CnnMlpPreModel, self).__init__(spec, model)

        self.transform = T.Compose([
            T.ToTensor(),
        ])
        self._print_freq = 2
        self.parse_q = parse_q

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EPreDataset.from_baseline_dataset(self.spec, train_data,
                                                            transform=self.transform, parse_q=self.parse_q)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EPreDataset.from_baseline_dataset(self.spec, eval_data,
                                                           transform=self.transform, parse_q=self.parse_q)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        feats = torch.stack(frames).to(self._device)
        if self.parse_q:
            qs = torch.stack(questions).to(self._device)
        else:
            qs = pack_sequence(questions, enforce_sorted=False).to(self._device)

        return feats, qs

    def _set_hyperparams(self):
        epochs = 20
        lr = 0.001
        batch_size = 256
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, parse_q=False):
        network = CnnMlpPreNetwork(spec, parse_q=parse_q)
        model = CnnMlpPreModel(spec, network, parse_q=parse_q)
        return model

    @staticmethod
    def load(spec, path, parse_q=False):
        model_path = Path(path) / "network.pt"
        network = util.load_model(CnnMlpPreNetwork, model_path, spec, parse_q)
        model = CnnMlpPreModel(spec, network, parse_q=parse_q)
        return model


# ********************************************************************************************
# ************************************ Object Models *****************************************
# ********************************************************************************************


class CnnObjModel(_AbsNeuralModel):
    def __init__(self, spec, model, parse_q=False, att=False):
        super(CnnObjModel, self).__init__(spec, model)

        self._print_freq = 10
        self.parse_q = parse_q
        self.att = att

        num_workers = os.cpu_count()
        self._future_timeout = 5
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
            qs = torch.stack(questions).to(self._device)
        else:
            qs = pack_sequence(questions, enforce_sorted=False).to(self._device)

        frames = [frame for frames_ in frames for frame in frames_]
        new_frames = frames

        if self.att:
            lens = []
            frame_objs = []
            for frame in frames:
                lens.append(len(frame))
                objs = [obj for obj, _ in frame]
                frame_objs.append(torch.stack(objs))

            frame_objs = pad_sequence(frame_objs)
            qs_pad, _ = pad_packed_sequence(qs)

            # Map questions to obj emb size and apply attention
            qs_pad = qs_pad.transpose(0, 1)
            qs_att = self._model.word_obj_map(qs_pad).transpose(0, 1).float()
            qs_att = qs_att.repeat_interleave(32, dim=1)

            frame_objs = frame_objs.float().to(self._device)
            frames_objs, _ = self._model.obj_att(frame_objs, qs_att, qs_att)
            frame_objs = list(frame_objs.cpu().transpose(0, 1))

            new_frames = []
            for f_idx, frame in enumerate(frames):
                att_frame = []
                for o_idx, (_, position) in enumerate(frame):
                    obj = frame_objs[f_idx][o_idx, :]
                    att_frame.append((obj, position))
                new_frames.append(att_frame)

        futures = [self._executor.submit(self._gen_object_frame, frame) for frame in new_frames]
        obj_frames = [future.result(self._future_timeout) for future in futures]
        obj_frames = torch.stack(obj_frames).to(self._device)

        return obj_frames, qs

    def _gen_object_frame(self, frame):
        obj_feats = len(frame[0][0])
        obj_frame = torch.zeros((obj_feats, 64, 64))
        for obj, pos in frame:
            (x1, y1, x2, y2) = tuple(map(int, pos))
            x1 = x1 // 4
            x2 = x2 // 4
            y1 = y1 // 4
            y2 = y2 // 4
            for x in range(x1, x2 + 1):
                for y in range(y1, y2 + 1):
                    obj_frame[:, y, x] = obj

        return obj_frame

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 64
        return epochs, lr, batch_size

    @staticmethod
    def new(spec, parse_q=False, att=False):
        network = CnnMlpObjNetwork(spec, parse_q=parse_q, att=att)
        model = CnnObjModel(spec, network, parse_q=parse_q, att=att)
        return model

    @staticmethod
    def load(spec, path, parse_q=False, att=False):
        model_path = Path(path) / "network.pt"
        network = util.load_model(CnnMlpObjNetwork, model_path, spec, parse_q, att)
        model = CnnObjModel(spec, network, parse_q=parse_q, att=att)
        return model


class PropRelObjModel(_AbsNeuralModel):
    def __init__(self, spec, model):
        super(PropRelObjModel, self).__init__(spec, model)

    def _prepare_train_data(self, train_data):
        fn = util.collate_func
        train_dataset = E2EObjFilterDataset.from_video_dataset(self.spec, train_data, filter_qs=[0, 1], parse_q=True)
        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=fn)
        return train_loader

    def _prepare_eval_data(self, eval_data):
        eval_dataset = E2EObjFilterDataset.from_video_dataset(self.spec, eval_data, filter_qs=[0, 1], parse_q=True)
        eval_loader = DataLoader(eval_dataset, batch_size=self._batch_size, shuffle=True, collate_fn=util.collate_func)
        return eval_loader

    def _prepare_input(self, frames, questions, q_types, answers):
        objs = [torch.stack(objs) for objs in frames]
        objs = pad_sequence(objs).to(self._device)
        qs = torch.stack(questions)[None, :, :].to(self._device)
        return objs, qs

    def _set_hyperparams(self):
        epochs = 10
        lr = 0.001
        batch_size = 256
        return epochs, lr, batch_size

    @staticmethod
    def new(spec):
        network = PropRelObjNetwork(spec)
        model = PropRelObjModel(spec, network)
        return model

    @staticmethod
    def load(spec, path):
        model_path = Path(path) / "network.pt"
        network = util.load_model(PropRelObjNetwork, model_path, spec)
        model = PropRelObjModel(spec, network)
        return model
