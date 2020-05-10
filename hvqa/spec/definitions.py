# *** File common definitions ***


import torch
import torchvision.transforms as T


IMG_SIZE = 256
VIDEO_LENGTH = 32

DTYPE = torch.float32


# *** Properties ***

CLASSES = ["octopus", "fish", "bag", "rock"]


# *** Relations ***

CLOSE_TO = 5


# *** Transforms ***

detector_transforms = T.Compose([
    T.ToTensor(),
])
