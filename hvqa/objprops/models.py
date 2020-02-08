import torch
import torch.nn as nn


class PropertyExtractionModel(nn.Module):
    def __init__(self):
        super(PropertyExtractionModel, self).__init__()

        input_size = 16
        output_size = input_size // 2

        num_colours = 7
        num_rotations = 4

        self.relu = nn.ReLU()

        # Feature layers
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc = nn.Linear(64 * output_size * output_size, 1024)  # TODO dropout

        # Property layers
        self.colour_layer = nn.Linear(1024, num_colours)
        self.rotation_layer = nn.Linear(1024, num_rotations)

    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.norm(features)
        features = self.conv2(features)
        features = self.fc(self._flatten(features))
        features = self.relu(features)

        colours = self.colour_layer(features)
        colours = torch.softmax(colours, 0)

        rotations = self.rotation_layer(features)
        rotations = torch.softmax(rotations, 0)

        return colours, rotations

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)
