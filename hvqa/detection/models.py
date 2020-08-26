import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from hvqa.spec.definitions import IMG_SIZE


# Include background as a class
NUM_CLASSES = 4 + 1


class DetectionModel(nn.Module):
    def __init__(self, backbone):
        super(DetectionModel, self).__init__()

        sizes = ((4, 8, 16),)
        ratios = ((0.5, 1, 2.0),)
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=-1)

        self.f_rcnn = FasterRCNN(backbone,
                                 min_size=IMG_SIZE,
                                 max_size=IMG_SIZE,
                                 num_classes=NUM_CLASSES,
                                 rpn_anchor_generator=anchor_generator,
                                 box_roi_pool=roi_pooler,
                                 image_mean=(0.2514, 0.7528, 0.7001),
                                 image_std=(0.0692, 0.0521, 0.0446))

    def forward(self, x, target=None):
        return self.f_rcnn(x, target)


class ClassifierModel(nn.Module):
    def __init__(self):
        super(ClassifierModel, self).__init__()

        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=4)
        self.out_channels = 256

    def forward(self, img):
        resnet_out = self.resnet(img)
        out = torch.sigmoid(resnet_out)

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


class DetectionBackboneWrapper(nn.Module):
    def __init__(self, trained_model):
        super(DetectionBackboneWrapper, self).__init__()

        self.model = trained_model
        self.out_channels = trained_model.out_channels

        self.conv1 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.model.feature_map(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DetectionBackbone(nn.Module):
    def __init__(self):
        super(DetectionBackbone, self).__init__()

        in_channels = 3
        feat1 = 32
        feat2 = 64
        feat3 = 128

        self.conv1 = nn.Conv2d(in_channels, feat1, 3, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(feat1)
        self.conv2 = nn.Conv2d(feat1, feat2, 3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(feat2)
        self.conv3 = nn.Conv2d(feat2, feat3, 3, stride=2, padding=1)

        self.out_channels = feat3

    def forward(self, img):
        out = self.conv1(img)
        out = F.relu(out, inplace=True)
        out = self.norm1(out)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.norm2(out)
        out = self.conv3(out)
        out = F.relu(out, inplace=True)

        return out
