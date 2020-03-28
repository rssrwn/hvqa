# *** File common definitions ***


import torch
import torchvision.transforms as T


IMG_SIZE = 256
VIDEO_LENGTH = 32

DTYPE = torch.float32


# *** Properties ***

PROPERTIES = ["colour", "rotation", "class"]
COLOURS = ["red", "silver", "white", "brown", "blue", "purple", "green"]
ROTATIONS = [0, 1, 2, 3]
CLASSES = ["octopus", "fish", "bag", "rock"]
PROPS_ARR = [COLOURS, ROTATIONS, CLASSES]


# *** Relations ***

CLOSE_TO = 5

RELATIONS = ["close"]


# *** Events ****

EVENTS = ["move", "rotate_left", "rotate_right"]


# *** Transforms ***

detector_transforms = T.Compose([
    T.ToTensor(),
])

prop_transforms = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])
