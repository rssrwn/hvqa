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


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- Object Networks ---------------------------------------------
# ------------------------------------------------------------------------------------------------------


class CnnMlpObjNetwork(nn.Module):
    def __init__(self, spec, parse_q=False):
        super(CnnMlpObjNetwork, self).__init__()

        self.parse_q = parse_q

        obj_feat_size = 16 + 4 + 20
        feat_output_size = 256
        word_vector_size = 300
        hidden_size = 1024
        num_lstm_layers = 2

        parsed_q_input = 260
        parsed_q_feat_1 = 4096
        parsed_q_feat_2 = 2048

        dropout = 0.2
        mlp_input = (32 * feat_output_size) + hidden_size
        feat1 = 4096
        feat2 = 512

        if parse_q:
            self.lang_enc = nn.Sequential(
                nn.Linear(parsed_q_input, parsed_q_feat_1),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(parsed_q_feat_1, parsed_q_feat_2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(parsed_q_feat_2, hidden_size),
                nn.ReLU(),
            )
        else:
            self.lang_enc = _QuestionNetwork(word_vector_size, hidden_size, num_layers=num_lstm_layers)

        self.feat_extr = _MedFeatExtrNetwork(obj_feat_size, feat_output_size)
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
        q_feats = self.lang_enc(qs)

        v_feats = frame_feats.reshape((batch_size, -1))
        video_enc = torch.cat([v_feats, q_feats], dim=1)
        output = self.mlp(video_enc)

        return output


class TvqaNetwork(nn.Module):
    def __init__(self, spec):
        super(TvqaNetwork, self).__init__()

        self.obj_att_stream = _TvqaObjAttStream(spec)
        self.event_stream = _TvqaEventStream(spec)
        self._log_sm = nn.LogSoftmax(dim=1)

    def forward(self, x):
        obj_frames, frame_pairs, qs = x

        obj_att_prediction = self.obj_att_stream((obj_frames, qs))
        event_prediction = self.event_stream((frame_pairs, qs))

        output = {}
        for q_type, obj_att_preds in obj_att_prediction.items():
            event_preds = event_prediction[q_type]
            preds = obj_att_preds + event_preds
            preds = self._log_sm(preds)
            output[q_type] = preds

        return output


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- Helper Networks ---------------------------------------------
# ------------------------------------------------------------------------------------------------------


class _TvqaObjAttStream(nn.Module):
    def __init__(self, spec):
        super(_TvqaObjAttStream, self).__init__()

        obj_size = 20 + 16 + 4 + 4
        obj_enc_size = 128

        q_size = 260
        q_enc_size = obj_enc_size * 2
        frame_enc_size = obj_enc_size * 2

        num_att_heads = 8

        dropout = 0.2
        mlp_feat1 = 256
        mlp_feat2 = 128

        self.obj_fc = nn.Sequential(
            nn.Linear(obj_size, obj_enc_size),
            nn.ReLU()
        )
        self.q_fc = nn.Sequential(
            nn.Linear(q_size, q_enc_size),
            nn.ReLU()
        )
        self.frame_enc = _ObjEncNetwork(obj_enc_size, q_enc_size)
        self.frames_lstm = nn.LSTM(obj_enc_size, obj_enc_size, bidirectional=True)
        self.att = nn.MultiheadAttention(q_enc_size, num_att_heads)

        self.video_lstm = nn.LSTM(frame_enc_size * 3, frame_enc_size, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(frame_enc_size * 2, mlp_feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat1, mlp_feat2),
            nn.ReLU(),
            _QANetwork(spec, mlp_feat2, apply_sm=False)
        )

    def forward(self, x):
        frames, qs = x

        frames = self.obj_fc(frames)
        qs = self.q_fc(qs).unsqueeze(0).repeat_interleave(32, dim=1)
        frame_encs = self.frame_enc((frames, qs))

        batch_size = frame_encs.shape[0] // 32
        frame_encs = frame_encs.reshape((32, batch_size, -1))
        frame_encs, _ = self.frames_lstm(frame_encs)
        frame_encs_att, _ = self.att(frame_encs, qs, qs)

        enc = frame_encs * frame_encs_att
        enc = torch.cat([frame_encs, frame_encs_att, enc], dim=2)
        enc, _ = self.video_lstm(enc)
        enc, _ = torch.max(enc, dim=0)

        output = self.mlp(enc)
        return output


class _TvqaEventStream(nn.Module):
    def __init__(self, spec):
        super(_TvqaEventStream, self).__init__()

        enc_size = 128

        q_size = 260
        q_enc_size = enc_size * 2
        frame_enc_size = enc_size * 2

        num_att_heads = 8

        dropout = 0.2
        mlp_feat1 = 256
        mlp_feat2 = 128

        self.event_enc = _EventEncNetwork(enc_size)
        self.q_fc = nn.Sequential(
            nn.Linear(q_size, q_enc_size),
            nn.ReLU()
        )

        self.frames_lstm = nn.LSTM(enc_size, enc_size, bidirectional=True)
        self.att = nn.MultiheadAttention(q_enc_size, num_att_heads)

        self.video_lstm = nn.LSTM(frame_enc_size * 3, frame_enc_size, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(frame_enc_size * 2, mlp_feat1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat1, mlp_feat2),
            nn.ReLU(),
            _QANetwork(spec, mlp_feat2, apply_sm=False)
        )

    def forward(self, x):
        frames, qs = x

        frame_encs = self.event_enc(frames)
        qs = self.q_fc(qs).unsqueeze(0)

        batch_size = frame_encs.shape[0] // 31
        frame_encs = frame_encs.reshape((31, batch_size, -1))
        frame_encs, _ = self.frames_lstm(frame_encs)
        frame_encs_att, _ = self.att(frame_encs, qs, qs)

        enc = frame_encs * frame_encs_att
        enc = torch.cat([frame_encs, frame_encs_att, enc], dim=2)
        enc, _ = self.video_lstm(enc)
        enc, _ = torch.max(enc, dim=0)

        output = self.mlp(enc)
        return output


class _EventEncNetwork(nn.Module):
    def __init__(self, enc_size):
        super(_EventEncNetwork, self).__init__()

        cnn_output = 256
        dropout = 0.2

        self.feat_extr = _MedFeatExtrNetwork(6, cnn_output)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_output, enc_size),
            nn.ReLU(),
        )

    def forward(self, x):
        frames = x
        frame_encs = self.feat_extr(frames)
        output = self.mlp(frame_encs)
        return output

    # def _enc_frames(self, objs, pos):
    #     b_enc = []
    #     for f_idx, frame in enumerate(pos):
    #         f_encs = []
    #         for o_idx, obj_pos in enumerate(frame):
    #             obj_enc = objs[o_idx, f_idx, :]
    #             f_encs.append((obj_enc, obj_pos))
    #         b_enc.append(f_encs)
    #
    #     frames_enc = util.gen_object_frames(b_enc, self._executor)
    #     frames = torch.stack(frames_enc).to(self._device)
    #     return frames


class _ObjEncNetwork(nn.Module):
    def __init__(self, obj_feat_size, q_enc_size):
        super(_ObjEncNetwork, self).__init__()

        num_att_heads = 8
        dropout = 0.2

        self.self_att_1 = nn.MultiheadAttention(obj_feat_size, num_att_heads)

        self.q_fc = nn.Linear(q_enc_size, obj_feat_size)
        self.q_att = nn.MultiheadAttention(obj_feat_size, num_att_heads)

        self.self_att_2 = nn.MultiheadAttention(obj_feat_size, num_att_heads)
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(obj_feat_size, obj_feat_size),
            nn.ReLU()
        )

    def forward(self, x):
        objs, qs = x

        objs, _ = self.self_att_1(objs, objs, objs)
        objs = torch.relu(objs)

        qs = self.q_fc(qs)
        qs = torch.relu(qs)

        objs, _ = self.q_att(objs, qs, qs)
        objs = torch.relu(objs)

        objs, _ = self.self_att_2(objs, objs, objs)
        objs = torch.relu(objs)

        enc, _ = torch.max(objs, dim=0)
        output = self.fc_out(enc)

        return output


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

        feat1 = 32
        feat2 = 64
        feat3 = 128

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
    def __init__(self, spec, vector_size, apply_sm=True):
        super(_QANetwork, self).__init__()

        num_colours = len(spec.prop_values("colour"))
        num_rotations = len(spec.prop_values("rotation"))
        num_actions = len(spec.actions)
        num_effects = len(spec.effects)
        num_frames = spec.num_frames
        num_mc_options_q_6 = 3

        self.q_0_layer = nn.Sequential(
            nn.Linear(vector_size, num_colours + num_rotations),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_1_layer = nn.Sequential(
            nn.Linear(vector_size, 2),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_2_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_3_layer = nn.Sequential(
            nn.Linear(vector_size, (num_colours * num_colours) + (num_rotations * num_rotations)),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_4_layer = nn.Sequential(
            nn.Linear(vector_size, num_frames - 1),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_5_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions + num_effects),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_6_layer = nn.Sequential(
            nn.Linear(vector_size, num_actions),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_7_layer = nn.Sequential(
            nn.Linear(vector_size, num_mc_options_q_6),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
        )
        self.q_8_layer = nn.Sequential(
            nn.Linear(vector_size, num_colours),
            nn.LogSoftmax(dim=1) if apply_sm else nn.Identity()
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
