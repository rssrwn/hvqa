import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


# Include background as a class
NUM_CLASSES = 4 + 1


class DetectionModel(nn.Module):
    def __init__(self, backbone):
        super(DetectionModel, self).__init__()

        sizes = ((4, 8, 16, 32),)
        ratios = ((0.5, 1, 2.0),)
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=3, sampling_ratio=2)

        self.f_rcnn = FasterRCNN(backbone,
                                 min_size=128,
                                 max_size=128,
                                 num_classes=NUM_CLASSES,
                                 rpn_anchor_generator=anchor_generator,
                                 box_roi_pool=roi_pooler)

    def forward(self, x, target=None):
        return self.f_rcnn(x, target)


class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()

        # self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
        # self.out_channels = 256

        self.conv1 = nn.Conv2d(4, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 16 * 16, 1024)
        self.fc2 = nn.Linear(1024, 4)

    def forward(self, img):
        # resnet_out = self.resnet(img)
        # out = torch.sigmoid(resnet_out)

        out = self.conv1(img)
        out = self.conv2(out)
        out = self.pool1(out)
        out = self._flatten(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out

    def feature_map(self, x):
        # Compute output from ResNet model
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)

        return x

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)


class DetectionBackbone(nn.Module):
    def __init__(self, trained_model):
        super(DetectionBackbone, self).__init__()

        self.model = trained_model
        self.out_channels = trained_model.out_channels

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)

        # Freeze ResNet params
        for param in trained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.feature_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
