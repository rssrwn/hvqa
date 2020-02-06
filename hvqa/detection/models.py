import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


OUTPUT_SHAPE = (9, 8, 8)
TOTAL_OUTPUT_SIZE = OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1] * OUTPUT_SHAPE[2]

# Include background as a class
NUM_CLASSES = 4 + 1


class DetectionModel(nn.Module):
    def __init__(self, backbone):
        super(DetectionModel, self).__init__()

        sizes = ((4, 8, 16, 32),)
        ratios = ((0.5, 1, 2.0),)
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=3, sampling_ratio=-1)

        self.f_rcnn = FasterRCNN(backbone,
                                 min_size=128,
                                 max_size=128,
                                 num_classes=NUM_CLASSES,
                                 rpn_anchor_generator=anchor_generator,
                                 box_roi_pool=roi_pooler)

    def forward(self, img, target=None):
        return self.f_rcnn(img, target)

    @staticmethod
    def _flatten(tensor):
        batch = tensor.shape[0]
        return tensor.view(batch, -1)

    @staticmethod
    def _output_tensor(vec):
        batch = vec.shape[0]
        c, h, w = OUTPUT_SHAPE
        return vec.view(batch, c, h, w)


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
