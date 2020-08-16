import os
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pad_packed_sequence

import hvqa.util.func as util


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- VideoQA Networks --------------------------------------------
# ------------------------------------------------------------------------------------------------------


class LangLstmNetwork(nn.Module):
    def __init__(self, spec):
        super(LangLstmNetwork, self).__init__()

        word_vector_size = 300
        hidden_size = 1024
        num_lstm_layers = 2
        feat1 = 512
        feat2 = 256

        self.network = nn.Sequential(
            _QuestionNetwork(word_vector_size, hidden_size, num_layers=num_lstm_layers),
            nn.ReLU(),
            nn.Linear(hidden_size, feat1),
            nn.ReLU(),
            nn.Linear(feat1, feat2),
            nn.ReLU(),
            _QANetwork(spec, feat2)
        )

    def forward(self, x):
        return self.network(x)


class CnnMlpNetwork(nn.Module):
    def __init__(self, spec):
        super(CnnMlpNetwork, self).__init__()

        feat_output_size = 256
        word_vector_size = 300
        hidden_size = 1024
        num_lstm_layers = 2

        mlp_input = (32 * feat_output_size) + hidden_size
        feat1 = 4096
        feat2 = 1024
        feat3 = 512

        dropout = 0.2

        self.feat_extr = _VideoFeatNetwork(feat_output_size)
        self.lang_lstm = _QuestionNetwork(word_vector_size, hidden_size, num_layers=num_lstm_layers)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat2, feat3),
            nn.ReLU(),
            _QANetwork(spec, feat3)
        )

    def forward(self, x):
        frames, qs = x
        frame_feats = self.feat_extr(frames)
        batch_size = frame_feats.shape[0] // 32
        q_feats = self.lang_lstm(qs)
        v_feats = frame_feats.reshape((batch_size, -1))
        video_enc = torch.cat([v_feats, q_feats], dim=1)
        output = self.mlp(video_enc)
        return output


class CnnLstmNetwork(nn.Module):
    def __init__(self, spec):
        super(CnnLstmNetwork, self).__init__()

        feat_output_size = 256
        word_vector_size = 300

        q_hidden_size = 1024
        q_layers = 2

        v_hidden_size = 1024
        v_layers = 2

        mlp_input = v_hidden_size + q_hidden_size
        feat1 = 1024
        feat2 = 512

        dropout = 0.2

        self.feat_extr = _VideoFeatNetwork(feat_output_size)
        self.video_lstm = _VideoLstmNetwork(feat_output_size, v_hidden_size, v_layers)
        self.lang_lstm = _QuestionNetwork(word_vector_size, q_hidden_size, num_layers=q_layers)

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(),
            _QANetwork(spec, feat2)
        )

    def forward(self, x):
        frames, qs = x

        frame_feats = self.feat_extr(frames)
        batch_size = frame_feats.shape[0] // 32
        frame_feats_ = frame_feats.reshape((32, batch_size, -1))
        v_feats = self.video_lstm(frame_feats_)

        q_feats = self.lang_lstm(qs)
        video_enc = torch.cat([v_feats, q_feats], dim=1)
        output = self.mlp(video_enc)

        return output


class CnnMlpPreNetwork(nn.Module):
    def __init__(self, spec, parse_q=False):
        super(CnnMlpPreNetwork, self).__init__()

        feat_output_size = (32 * 32) + (32 * 31)

        q_input_size = 260 if parse_q else 300
        q_hidden_size = 1024
        q_layers = 2
        self.q_enc = _QuestionNetwork(q_input_size, q_hidden_size, parse_q=parse_q, num_layers=q_layers)

        mlp_input = feat_output_size + q_hidden_size
        feat1 = 1024
        feat2 = 256
        dropout = 0.2
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            _QANetwork(spec, feat2)
        )

    def forward(self, x):
        feats, qs = x
        q_feats = self.q_enc(qs)
        video_enc = torch.cat([feats, q_feats], dim=1)
        output = self.mlp(video_enc)
        return output


class PropRelNetwork(nn.Module):
    def __init__(self, spec):
        super(PropRelNetwork, self).__init__()

        feat_output_size = 32
        self.feat_extr = _SmallFeatExtrNetwork(3, feat_output_size)

        word_vector_size = 300
        q_hidden_size = 1024
        q_layers = 2
        self.lang_lstm = _QuestionNetwork(word_vector_size, q_hidden_size, num_layers=q_layers)

        mlp_input = feat_output_size + q_hidden_size
        feat1 = 512
        dropout = 0.5
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(),
            _QANetwork(spec, feat1)
        )

    def forward(self, x):
        frames, qs = x

        frame_feats = self.feat_extr(frames)
        q_feats = self.lang_lstm(qs)

        feats = torch.cat((frame_feats, q_feats), dim=1)
        output = self.mlp(feats)

        return output


class EventNetwork(nn.Module):
    def __init__(self, spec):
        super(EventNetwork, self).__init__()

        feat_output_size = 32
        feat1 = 16
        dropout = 0.2

        self.feat_extr = _SmallFeatExtrNetwork(6, feat_output_size)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_output_size, feat1),
            nn.ReLU(),
            _QANetwork(spec, feat1)
        )

    def forward(self, x):
        feats = self.feat_extr(x)
        output = self.mlp(feats)
        return output


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- Object Networks ---------------------------------------------
# ------------------------------------------------------------------------------------------------------


class CnnMlpObjNetwork(nn.Module):
    def __init__(self, spec, parse_q=False, att=False):
        super(CnnMlpObjNetwork, self).__init__()

        self.parse_q = parse_q
        self.att = att

        obj_feat_size = 8 + 4 + 20 + 4
        feat_output_size = 256
        word_vector_size = 300
        hidden_size = 1024
        num_lstm_layers = 2

        mlp_input = (32 * feat_output_size) + hidden_size
        feat1 = 4096
        feat2 = 512

        dropout = 0.2

        num_att_heads = 4

        self.feat_extr = _MedFeatExtrNetwork(obj_feat_size, feat_output_size)
        self.lang_lstm = _QuestionNetwork(word_vector_size, hidden_size, num_layers=num_lstm_layers)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(),
            _QANetwork(spec, feat2)
        )

        if att:
            self.word_obj_map = nn.Linear(word_vector_size, obj_feat_size)
            self.obj_att = nn.MultiheadAttention(obj_feat_size, num_att_heads)
            self.word_frame_map = nn.Linear(word_vector_size, feat_output_size)
            self.frame_att = nn.MultiheadAttention(feat_output_size, num_att_heads)

    def forward(self, x):
        frames, qs = x
        frame_feats = self.feat_extr(frames)
        batch_size = frame_feats.shape[0] // 32
        q_feats = self.lang_lstm(qs)

        if self.att:
            # Prepare question attention input
            qs_att, _ = pad_packed_sequence(qs)
            qs_att = qs_att.transpose(0, 1)
            qs_att = self.word_frame_map(qs_att).transpose(0, 1)

            v_feats = frame_feats.reshape((32, batch_size, -1))
            v_feats, _ = self.frame_att(v_feats, qs_att, qs_att)
            frame_feats = v_feats.transpose(0, 1)

        v_feats = frame_feats.reshape((batch_size, -1))
        video_enc = torch.cat([v_feats, q_feats], dim=1)
        output = self.mlp(video_enc)
        return output


class PropRelObjNetwork(nn.Module):
    def __init__(self, spec):
        super(PropRelObjNetwork, self).__init__()

        num_att_heads = 8
        obj_enc_size = 20 + 16 + 4 + 4
        obj_feat_size = 128

        q_enc_size = 260

        mlp_feat1 = 64
        dropout = 0.5

        self.obj_fc = nn.Linear(obj_enc_size, obj_feat_size)
        self.self_att_1 = nn.MultiheadAttention(obj_feat_size, num_att_heads)

        self.q_fc = nn.Linear(q_enc_size, obj_feat_size)
        self.q_att = nn.MultiheadAttention(obj_feat_size, num_att_heads)

        self.self_att_2 = nn.MultiheadAttention(obj_feat_size, num_att_heads)

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(obj_feat_size, mlp_feat1),
            nn.ReLU(),
            _QANetwork(spec, mlp_feat1)
        )

    def forward(self, x):
        objs, qs = x

        objs = self.obj_fc(objs)
        objs = torch.relu(objs)

        objs, _ = self.self_att_1(objs, objs, objs)
        objs = torch.relu(objs)

        qs = self.q_fc(qs)
        qs = torch.relu(qs)

        objs, _ = self.q_att(objs, qs, qs)
        objs = torch.relu(objs)

        objs, _ = self.self_att_2(objs, objs, objs)
        objs = torch.relu(objs)

        enc, _ = torch.max(objs, dim=0)
        output = self.mlp(enc)

        return output


class EventObjNetwork(nn.Module):
    def __init__(self, spec):
        super(EventObjNetwork, self).__init__()

        num_workers = os.cpu_count()
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._device = util.get_device()

        obj_enc_size = 20 + 16 + 4 + 4
        obj_feat_size = 16

        cnn_output = 32
        dropout = 0.5
        mlp_feat1 = 16

        self.obj_fc = nn.Linear(obj_enc_size, obj_feat_size)
        self.feat_extr = _SmallFeatExtrNetwork(obj_feat_size * 2, cnn_output)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_output, mlp_feat1),
            nn.ReLU(),
            _QANetwork(spec, mlp_feat1)
        )

    def forward(self, x):
        (objs, objs_pos), (next_objs, next_objs_pos) = x

        objs = self.obj_fc(objs)
        objs = torch.relu(objs)

        next_objs = self.obj_fc(next_objs)
        next_objs = torch.relu(next_objs)

        frames = self._enc_frames(objs, objs_pos)
        next_frames = self._enc_frames(next_objs, next_objs_pos)
        v_enc = torch.cat([frames, next_frames], dim=1)
        enc = self.feat_extr(v_enc)

        output = self.mlp(enc)

        return output

    def _enc_frames(self, objs, pos):
        b_enc = []
        for f_idx, frame in enumerate(pos):
            f_encs = []
            for o_idx, obj_pos in enumerate(frame):
                obj_enc = objs[o_idx, f_idx, :]
                f_encs.append((obj_enc, obj_pos))
            b_enc.append(f_encs)

        frames_enc = util.gen_object_frames(b_enc, self._executor)
        frames = torch.stack(frames_enc).to(self._device)
        return frames


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- Helper Networks ---------------------------------------------
# ------------------------------------------------------------------------------------------------------


class _SmallFeatExtrNetwork(nn.Module):
    def __init__(self, in_channels, output_size):
        super(_SmallFeatExtrNetwork, self).__init__()

        feat1 = 8
        feat2 = 16
        feat3 = 32

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, feat1, kernel_size=3),
            nn.BatchNorm2d(feat1),
            nn.ReLU(),
            nn.Conv2d(feat1, feat2, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat2),
            nn.ReLU(),
            nn.Conv2d(feat2, feat3, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(feat3, output_size)
        )

    def forward(self, x):
        feats = self.network(x)
        return feats


class _MedFeatExtrNetwork(nn.Module):
    def __init__(self, in_channels, output_size):
        super(_MedFeatExtrNetwork, self).__init__()

        feat1 = 64
        feat2 = 128
        feat3 = 256

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, feat1, kernel_size=3),
            nn.BatchNorm2d(feat1),
            nn.ReLU(),
            nn.Conv2d(feat1, feat2, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat2),
            nn.ReLU(),
            nn.Conv2d(feat2, feat3, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(feat3, output_size)
        )

    def forward(self, x):
        feats = self.network(x)
        return feats


class _VideoLstmNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(_VideoLstmNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        out = h_n[-1, :, :]
        return out


class _VideoFeatNetwork(nn.Module):
    def __init__(self, output_size, in_channels=3):
        super(_VideoFeatNetwork, self).__init__()

        resnet_out = 512

        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(resnet_out, output_size)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.resnet(x)


class _QuestionNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, parse_q=False, num_layers=2):
        super(_QuestionNetwork, self).__init__()

        self.parse_q = parse_q

        if parse_q:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers)

    def forward(self, x):
        if self.parse_q:
            out = self.mlp(x)
        else:
            _, (h_n, c_n) = self.lstm(x)
            out = h_n[-1, :, :]

        return out


class _QANetwork(nn.Module):
    def __init__(self, spec, vector_size):
        super(_QANetwork, self).__init__()

        num_colours = len(spec.prop_values("colour"))
        num_rotations = len(spec.prop_values("rotation"))
        num_actions = len(spec.actions)
        num_effects = len(spec.effects)
        num_frames = spec.num_frames
        num_mc_options_q_6 = 3

        self.q_0_layer = nn.Sequential(
            nn.Linear(vector_size, num_colours + num_rotations),
            nn.LogSoftmax(dim=1)
        )
        self.q_1_layer = nn.Sequential(
            nn.Linear(vector_size, 2),
            nn.LogSoftmax(dim=1)
        )
        self.q_2_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions),
            nn.LogSoftmax(dim=1)
        )
        self.q_3_layer = nn.Sequential(
            nn.Linear(vector_size, (num_colours * num_colours) + (num_rotations * num_rotations)),
            nn.LogSoftmax(dim=1)
        )
        self.q_4_layer = nn.Sequential(
            nn.Linear(vector_size, num_frames - 1),
            nn.LogSoftmax(dim=1)
        )
        self.q_5_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions + num_effects),
            nn.LogSoftmax(dim=1)
        )
        self.q_6_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions),
            nn.LogSoftmax(dim=1)
        )
        self.q_7_layer = nn.Sequential(
            nn.Linear(vector_size, num_mc_options_q_6),
            nn.LogSoftmax(dim=1)
        )
        self.q_8_layer = nn.Sequential(
            nn.Linear(vector_size, num_colours),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        q_0 = self.q_0_layer(x)
        q_1 = self.q_1_layer(x)
        q_2 = self.q_2_layer(x)
        q_3 = self.q_3_layer(x)
        q_4 = self.q_4_layer(x)
        q_5 = self.q_5_layer(x)
        q_6 = self.q_6_layer(x)
        q_7 = self.q_7_layer(x)
        q_8 = self.q_8_layer(x)

        output = {
            0: q_0,
            1: q_1,
            2: q_2,
            3: q_3,
            4: q_4,
            5: q_5,
            6: q_6,
            7: q_7,
            8: q_8
        }

        return output
