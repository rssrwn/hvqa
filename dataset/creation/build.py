import random

import numpy as np
import cv2


def build_frames(num_frames):
    template, octopus = initial_frame()
    initial = { "objects": template["objects"] + octopus }

    frames = [initial]
    for frame in range(1, num_frames):
        next, octopus = move_octopus(template, octopus)

    # TODO
