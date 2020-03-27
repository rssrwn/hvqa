# *** File common definitions ***


import torch
import torchvision.transforms as T


IMG_SIZE = 256

DTYPE = torch.float32

PROPERTIES = ["colour", "rotation", "class"]
COLOURS = ["red", "silver", "white", "brown", "blue", "purple", "green"]
ROTATIONS = [0, 1, 2, 3]
CLASSES = ["octopus", "fish", "bag", "rock"]
PROPS_ARR = [COLOURS, ROTATIONS, CLASSES]

detector_transforms = T.Compose([
    T.ToTensor(),
])

prop_transforms = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])
