import torch.nn as nn
import torch.nn.functional as F


OUTPUT_SHAPE = (9, 8, 8)


class DetectionModel(nn.module):
    def __init__(self):
        super().__init__()

        self.leaky_slope = 0.1

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(8 * 8 * 128, 1024)
        self.fc2 = nn.Linear(1024, 8 * 8 * 9)

    def forward(self, img):
        out = F.leaky_relu(self.conv1(img), self.leaky_slope)
        out = F.leaky_relu(self.conv2(out), self.leaky_slope)
        out = self.pool1(out)

        out = F.leaky_relu(self.conv3(out), self.leaky_slope)
        out = F.leaky_relu(self.conv4(out), self.leaky_slope)
        out = self.pool2(out)

        out = F.leaky_relu(self.conv5(out), self.leaky_slope)
        out = F.leaky_relu(self.conv6(out), self.leaky_slope)
        out = self.pool3(out)

        out = self.fc1(self._flatten(out))
        out = self.fc2(out)
        return self._output_tensor(out)

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)

    @staticmethod
    def _output_tensor(vec):
        batch = vec[0]
        c, h, w = OUTPUT_SHAPE
        return vec.view(batch, c, h, w)
