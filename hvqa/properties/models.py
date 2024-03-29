import torch
import torch.nn as nn
import torch.nn.functional as F


class PropertyExtractionModel(nn.Module):
    def __init__(self, spec):
        """
        Create a torch Module for the properties model

        :param spec: EnvSpec obj
        """

        super(PropertyExtractionModel, self).__init__()

        output_size = 6

        feat1 = 32
        feat2 = 64
        feat3 = 64
        latent_size = 1024

        # Feature layers
        self.conv1 = nn.Conv2d(3, feat1, 3, stride=1)
        self.norm1 = nn.BatchNorm2d(feat1)
        self.conv2 = nn.Conv2d(feat1, feat2, 3, stride=2)
        self.norm2 = nn.BatchNorm2d(feat2)
        self.conv3 = nn.Conv2d(feat2, feat3, 3, stride=2)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(feat3 * output_size * output_size, latent_size)

        # Property layers
        prop_layers = {}
        for prop in spec.prop_names():
            num_values = len(spec.prop_values(prop))
            prop_layer = nn.Linear(latent_size, num_values)
            self.add_module(prop, prop_layer)
            prop_layers[prop] = prop_layer

        self.prop_layers = prop_layers

    def forward(self, x):
        features = self.conv1(x)
        features = F.relu(features)
        features = self.norm1(features)
        features = self.conv2(features)
        features = F.relu(features)
        features = self.norm2(features)
        features = self.conv3(features)
        features = F.relu(features)

        features = self._flatten(features)
        features = self.drop(features)
        features = self.fc(features)
        features = F.relu(features)

        prop_outs = {prop: layer(features) for prop, layer in self.prop_layers.items()}

        # Note: softmax is applied by the loss function
        return prop_outs

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)


class ObjectAutoEncoder(nn.Module):
    def __init__(self, spec):
        super(ObjectAutoEncoder, self).__init__()

        self.spec = spec

        feat1 = 8
        feat2 = 16
        latent_size = 16

        self.feat2 = feat2

        self.conv1 = nn.Conv2d(3, feat1, 3, stride=1)
        self.norm1 = nn.BatchNorm2d(feat1)
        self.conv2 = nn.Conv2d(feat1, feat2, 3, stride=2)
        self.norm2 = nn.BatchNorm2d(feat2)
        self.fc1 = nn.Linear(6 * 6 * feat2, latent_size)

        self.fc2 = nn.Linear(latent_size, 6 * 6 * feat2)
        self.norm3 = nn.BatchNorm2d(feat2)
        self.trans_conv1 = nn.ConvTranspose2d(feat2, feat1, 3, stride=2, output_padding=1)
        self.norm4 = nn.BatchNorm2d(feat1)
        self.trans_conv2 = nn.ConvTranspose2d(feat1, 3, 3)

    def encode(self, x):
        features = self.conv1(x)
        features = F.relu(features)
        features = self.norm1(features)
        features = self.conv2(features)
        features = F.relu(features)
        features = self.norm2(features)
        features = self._flatten(features)
        latent = self.fc1(features)
        return latent

    def decode(self, x):
        features = self.fc2(x)
        features = F.relu(features)
        features = self._shape(features, self.feat2, 6)
        features = self.norm3(features)
        features = self.trans_conv1(features)
        features = F.relu(features)
        features = self.norm4(features)
        features = self.trans_conv2(features)
        out = torch.sigmoid(features)
        return out

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)

    @staticmethod
    def _shape(tensor, channels, size):
        batch = tensor.shape[0]
        return tensor.view(batch, channels, size, size)
