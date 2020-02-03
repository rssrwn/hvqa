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

        # self.leaky_slope = 0.01
        #
        # # Setup model
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        # self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        # self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        # self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        #
        # self.fc1 = nn.Linear(8 * 8 * 256, 2048)
        # self.fc2 = nn.Linear(2048, TOTAL_OUTPUT_SIZE)
        #
        # # Weight initialisation
        # nn.init.kaiming_normal_(self.conv1.weight)
        # nn.init.kaiming_normal_(self.conv2.weight)
        # nn.init.kaiming_normal_(self.conv3.weight)
        # nn.init.kaiming_normal_(self.conv4.weight)
        # nn.init.kaiming_normal_(self.conv5.weight)
        # nn.init.kaiming_normal_(self.conv6.weight)
        # nn.init.kaiming_normal_(self.conv7.weight)
        # nn.init.kaiming_normal_(self.conv8.weight)
        # nn.init.kaiming_normal_(self.fc1.weight)
        # nn.init.kaiming_normal_(self.fc2.weight)

        sizes = ((4, 8, 16),)
        ratios = ((0.5, 1, 2.0),)
        anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=['0'], output_size=[7], sampling_ratio=2)

        self.f_rcnn = FasterRCNN(backbone,
                                 min_size=128,
                                 max_size=128,
                                 num_classes=NUM_CLASSES,
                                 rpn_anchor_generator=anchor_generator,
                                 box_roi_pool=roi_pooler)

    def forward(self, x):
        # out = F.leaky_relu(self.conv1(img), self.leaky_slope)
        # out = F.leaky_relu(self.conv2(out), self.leaky_slope)
        # out = self.pool1(out)
        #
        # out = F.leaky_relu(self.conv3(out), self.leaky_slope)
        # out = F.leaky_relu(self.conv4(out), self.leaky_slope)
        # out = self.pool2(out)
        #
        # out = F.leaky_relu(self.conv5(out), self.leaky_slope)
        # out = F.leaky_relu(self.conv6(out), self.leaky_slope)
        # out = self.pool3(out)
        #
        # out = F.leaky_relu(self.conv7(out), self.leaky_slope)
        # out = F.leaky_relu(self.conv8(out), self.leaky_slope)
        # out = self.pool4(out)
        #
        # out = self.fc1(self._flatten(out))
        # out = F.relu(out)
        # out = self.fc2(out)
        # out = F.relu(out)
        # return self._output_tensor(out)

        return self.f_rcnn(x)

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
        self.out_channels = 512

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

        for param in trained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model.feature_map(x)
        return x
