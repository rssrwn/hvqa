import torch
import torch.nn as nn
import torchvision.models as models


# ------------------------------------------------------------------------------------------------------
# ---------------------------------------- VideoQA Networks --------------------------------------------
# ------------------------------------------------------------------------------------------------------


class LangLstmNetwork(nn.Module):
    def __init__(self, spec):
        super(LangLstmNetwork, self).__init__()

        word_vector_size = 300
        hidden_size = 512
        feat1 = 512
        feat2 = 256

        dropout = 0.2

        self.lstm = nn.LSTM(word_vector_size, hidden_size, bidirectional=True)

        self.network = nn.Sequential(
            nn.Linear(hidden_size * 2, feat1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(inplace=True),
            _QANetwork(spec, feat2)
        )

    def forward(self, x):
        _, (enc, _) = self.lstm(x)
        batch_size = enc.shape[1]
        enc = enc.transpose(0, 1).reshape(batch_size, -1)
        return self.network(enc)


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

        # self.feat_extr = _VideoFeatNetwork(feat_output_size)
        # self.lang_lstm = _QuestionNetwork(word_vector_size, hidden_size, num_layers=num_lstm_layers)

        self.feat_extr = _SmallFeatExtrNetwork(3, 32)
        self.lang_lstm = nn.LSTM(word_vector_size, 256, bidirectional=True)

        mlp_input = (32 * 32) + (256 * 2)
        feat1 = 512
        feat2 = 256

        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(mlp_input, feat1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(inplace=True),
            _QANetwork(spec, feat2)
        )

    def forward(self, x):
        frames, qs = x
        frame_feats = self.feat_extr(frames)
        batch_size = frame_feats.shape[0] // 32

        # q_feats = self.lang_lstm(qs)
        _, (q_feats, _) = self.lang_lstm(qs)
        q_feats = q_feats.transpose(0, 1).reshape((batch_size, -1))

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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(inplace=True),
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


class Cnn3DMlpNetwork(nn.Module):
    def __init__(self, spec):
        super(Cnn3DMlpNetwork, self).__init__()

        q_emb_size = 300
        q_hidden_size = 512

        cnn_feat1 = 32
        cnn_feat2 = 64
        cnn_feat3 = 128
        cnn_feat4 = 256

        dropout = 0.2
        video_enc_size = 1024

        mlp_feat1 = 1024
        mlp_feat2 = 512
        mlp_feat3 = 256

        self.q_enc = _QuestionNetwork(q_emb_size, q_hidden_size, num_layers=2)
        self.video_enc = nn.Sequential(
            nn.Conv3d(3, cnn_feat1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(cnn_feat1),
            nn.Conv3d(cnn_feat1, cnn_feat2, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(cnn_feat2),
            nn.Conv3d(cnn_feat2, cnn_feat3, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(cnn_feat3),
            nn.Conv3d(cnn_feat3, cnn_feat4, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(cnn_feat4),
            nn.AdaptiveMaxPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(cnn_feat4, video_enc_size),
            nn.ReLU(inplace=True)
        )
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(video_enc_size + q_hidden_size, mlp_feat1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat1, mlp_feat2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat2, mlp_feat3),
            nn.ReLU(inplace=True),
            _QANetwork(spec, mlp_feat3)
        )

    def forward(self, x):
        videos, qs = x

        q_enc = self.q_enc(qs)
        video_enc = self.video_enc(videos)

        enc = torch.cat([video_enc, q_enc], dim=1)
        output = self.mlp(enc)

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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat1, feat2),
            nn.ReLU(inplace=True),
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
# ---------------------------------------- MAC Networks -----------------------------------------------
# ------------------------------------------------------------------------------------------------------


class MacNetwork(nn.Module):
    def __init__(self, spec, p):
        super(MacNetwork, self).__init__()

        assert p >= 1, "Number of MAC cells must be at least 1"

        q_enc_size = 300
        hidden_size = 256

        self.q_enc = _MacQuestionEnc(q_enc_size, hidden_size)
        self.video_enc = _MacVideoEnc(hidden_size)

        self.mac_cells = nn.ModuleList()
        for _ in range(p):
            self.mac_cells.append(_MacCell(hidden_size))

        c0 = torch.randn((1, hidden_size), requires_grad=True)
        m0 = torch.randn((1, hidden_size), requires_grad=True)
        self.c0 = nn.Parameter(c0, True)
        self.m0 = nn.Parameter(m0, True)

        self.output_unit = _MacOutputUnit(spec, hidden_size)

    def forward(self, x):
        videos, qs = x

        q, ctx_words = self.q_enc(qs)
        k = self.video_enc((videos, qs))

        batch_size = k.shape[1]

        ci = self.c0.repeat_interleave(batch_size, dim=0)
        mi = self.m0.repeat_interleave(batch_size, dim=0)

        for cell in self.mac_cells:
            ci, mi = cell(((ci, mi), (q, k, ctx_words)))

        output = self.output_unit((q, mi))

        return output


class _MacCell(nn.Module):
    def __init__(self, hidden_size):
        super(_MacCell, self).__init__()

        self.q_map = nn.Linear(hidden_size * 2, hidden_size)

        self.ctrl = _MacControlUnit(hidden_size)
        self.read = _MacReadUnit(hidden_size)
        self.write = _MacWriteUnit(hidden_size)

    def forward(self, x):
        (ci_1, mi_1), (q, k, ctx_words) = x

        qi = self.q_map(q)
        ci = self.ctrl((ci_1, qi, ctx_words))
        ri = self.read((mi_1, k, ci))
        mi = self.write((ri, mi_1, ci))

        return ci, mi


class _MacOutputUnit(nn.Module):
    def __init__(self, spec, hidden_size):
        super(_MacOutputUnit, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(inplace=True),
            _QANetwork(spec, hidden_size)
        )

    def forward(self, x):
        q, m_out = x

        enc = torch.cat([q, m_out], dim=1)
        output = self.mlp(enc)

        return output


class _MacControlUnit(nn.Module):
    def __init__(self, hidden_size):
        super(_MacControlUnit, self).__init__()

        self.cq_i_map = nn.Linear(hidden_size * 2, hidden_size)
        self.att = _MacAttention(hidden_size)

    def forward(self, x):
        ci_1, qi, ctx_words = x

        ctrl = torch.cat([qi, ci_1], dim=1)
        ctrl = self.cq_i_map(ctrl)
        ci = self.att((ctx_words, ctx_words, ctrl))
        return ci


class _MacReadUnit(nn.Module):
    def __init__(self, hidden_size):
        super(_MacReadUnit, self).__init__()

        self.memory_map = nn.Linear(hidden_size, hidden_size)
        self.knowledge_map = nn.Linear(hidden_size, hidden_size)

        self.i_k_map = nn.Linear(hidden_size * 2, hidden_size)
        self.att = _MacAttention(hidden_size)

    def forward(self, x):
        mi_1, k, ci = x

        i = mi_1 * k
        i_k = torch.cat([i, k], dim=2)
        i_k = self.i_k_map(i_k)
        out = self.att((k, i_k, ci))
        return out


class _MacWriteUnit(nn.Module):
    def __init__(self, hidden_size, self_att=False):
        super(_MacWriteUnit, self).__init__()

        assert not self_att, "This operation is not yet supported"

        self.memory_map = nn.Linear(hidden_size * 2, hidden_size)
        self.ctrl_map = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ri, mi_1, ci = x

        mi_prime = torch.cat([mi_1, ri], dim=1)
        mi_prime = self.memory_map(mi_prime)

        ci = self.ctrl_map(ci)
        mi = (ci * mi_1) + ((1 - ci) * mi_prime)

        return mi


class _MacAttention(nn.Module):
    def __init__(self, hidden_size):
        super(_MacAttention, self).__init__()

        self.compat_map = nn.Linear(hidden_size, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        out_seq, query_seq, ctrl = x

        compat = ctrl * query_seq
        compat = self.compat_map(compat).squeeze(2)
        compat = self.softmax(compat)
        out = compat.unsqueeze(2) * out_seq
        out = out.sum(dim=0)
        return out


class _MacQuestionEnc(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(_MacQuestionEnc, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=1)
        self.ctx_words_map = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        ctx_words, (q, _) = self.lstm(x)
        ctx_words = self.ctx_words_map(ctx_words)
        q = torch.cat([q[0, :, :], q[1, :, :]], dim=1)
        return q, ctx_words


class _MacVideoEnc(nn.Module):
    def __init__(self, hidden_size):
        super(_MacVideoEnc, self).__init__()

        self.feat_extr = _VideoFeatNetwork(hidden_size, in_channels=3)
        # self.feat_extr = _MedFeatExtrNetwork(in_channels=3, output_size=hidden_size)

    def forward(self, x):
        frames, qs = x

        video_feats = self.feat_extr(frames)
        batch_size = video_feats.shape[0] // 32
        video_feats = video_feats.reshape((32, batch_size, -1))

        return video_feats


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
            nn.ReLU(inplace=True)
        )
        self.q_fc = nn.Sequential(
            nn.Linear(q_size, q_enc_size),
            nn.ReLU(inplace=True)
        )
        self.frame_enc = _ObjEncNetwork(obj_enc_size, q_enc_size)
        self.frames_lstm = nn.LSTM(obj_enc_size, obj_enc_size, bidirectional=True)
        self.att = nn.MultiheadAttention(q_enc_size, num_att_heads)

        self.video_lstm = nn.LSTM(frame_enc_size * 3, frame_enc_size, bidirectional=True)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(frame_enc_size * 2, mlp_feat1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat1, mlp_feat2),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_feat1, mlp_feat2),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True)
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
            nn.ReLU(inplace=True),
            nn.Conv2d(feat1, feat2, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat2, feat3, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat3),
            nn.ReLU(inplace=True),
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
            nn.ReLU(inplace=True),
            nn.Conv2d(feat1, feat2, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat2, feat3, kernel_size=3, stride=2),
            nn.BatchNorm2d(feat3),
            nn.ReLU(inplace=True),
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
                nn.ReLU(inplace=True),
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
