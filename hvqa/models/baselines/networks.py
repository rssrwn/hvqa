import torch
import torch.nn as nn
import torch.nn.functional as F


class LangLstmNetwork(nn.Module):
    def __init__(self):
        super(LangLstmNetwork, self).__init__()

    def forward(self, x):
        pass


class QuestionNetwork(nn.Module):
    def __init__(self):
        super(QuestionNetwork, self).__init__()

    def forward(self, x):
        pass


class QANetwork(nn.Module):
    def __init__(self, spec, vector_size):
        super(QANetwork, self).__init__()

        num_colours = len(spec.prop_values("colour"))
        num_rotations = len(spec.prop_values("rotation"))
        num_actions = len(spec.actions)
        num_effects = len(spec.effects)
        num_frames = spec.num_frames

        self.q_0_layer = nn.Linear(vector_size, num_colours + num_rotations)
        self.q_1_layer = nn.Linear(vector_size, 1)
        self.q_2_layer = nn.Linear(vector_size, num_actions)
        self.q_3_layer = nn.Linear(vector_size, (num_colours * num_colours) + (num_rotations * num_rotations))
        self.q_4_layer = nn.Linear(vector_size, num_frames - 1)
        self.q_5_layer = nn.Linear(vector_size, num_actions + num_effects)
        self.q_6_layer = nn.Linear(vector_size, num_actions)

    def forward(self, x):
        q_0 = self.q_0_layer(x)
        q_1 = self.q_1_layer(x)
        q_2 = self.q_2_layer(x)
        q_3 = self.q_3_layer(x)
        q_4 = self.q_4_layer(x)
        q_5 = self.q_5_layer(x)
        q_6 = self.q_6_layer(x)

        output = {
            0: q_0,
            1: q_1,
            2: q_2,
            3: q_3,
            4: q_4,
            5: q_5,
            6: q_6
        }

        return output
