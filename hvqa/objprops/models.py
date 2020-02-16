import torch.nn as nn


class PropertyExtractionModel(nn.Module):
    def __init__(self):
        super(PropertyExtractionModel, self).__init__()

        input_size = 16
        output_size = input_size // 2

        num_colours = 7
        num_rotations = 4
        num_classes = 4

        self.relu = nn.ReLU()

        # Feature layers
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.norm = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc = nn.Linear(64 * output_size * output_size, 1024)  # TODO dropout

        # Property layers
        self.colour_layer = nn.Linear(1024, num_colours)
        self.rotation_layer = nn.Linear(1024, num_rotations)
        self.class_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        features = self.conv1(x)
        features = self.relu(features)
        features = self.norm(features)
        features = self.pool(features)
        features = self.conv2(features)
        features = self._flatten(features)
        features = self.fc(features)
        features = self.relu(features)

        colours = self.colour_layer(features)
        rotations = self.rotation_layer(features)
        classes = self.class_layer(features)

        # Note: softmax is applied by the loss function
        return colours, rotations, classes

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)
