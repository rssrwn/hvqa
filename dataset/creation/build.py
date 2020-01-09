import random

import numpy as np
import cv2


NUM_FRAMES = 5

FRAME_SIZE = 128
NUM_SEGMENTS = 4
SEGMENT_SIZE = FRAME_SIZE / NUM_SEGMENTS
EDGE = 5
OBJ_DIST = 5

ROTATIONS = [0,1,2,3]
ROCK_COLOURS = ["brown", "blue", "purple", "green"]

OCTOPUS = (11,9)
OCTOPUS_MIDDLE = (6,5)
FISH = (7,5)
FISH_MIDDLE = (4,3)
BAG = (5,7)
BAG_MIDDLE = (3,4)
ROCK = (7,7)
ROCK_MIDDLE = (4,4)

MIN_OBJ = 2
MAX_OBJ = 4


def build_frames(num_frames):
    template, octopus = initial_frame()
    initial = { "objects": template["objects"] + octopus }

    frames = [initial]
    for frame in range(1, num_frames):
        next, octopus = move_octopus(template, octopus)

    # TODO


def place_obj(obj_size, rotation, used_segments):
    x_seg = random.randint(0, NUM_SEGMENTS)
    y_seg = random.randint(0, NUM_SEGMENTS)

    width = obj_size[0]
    height = obj_size[1]
    if rotation == 1 or rotation == 3:
        width = obj_size[1]
        height = obj_size[0]

    while (x_seg, y_seg) in used_segments:
        x_seg = random.randint(0, NUM_SEGMENTS)
        y_seg = random.randint(0, NUM_SEGMENTS)

    obj_x = random.randint(x_seg * SEGMENT_SIZE, ((x_seg + 1) * SEGMENT_SIZE) - width)
    obj_y = random.randint(y_seg * SEGMENT_SIZE, ((y_seg + 1) * SEGMENT_SIZE) - height)

    return [obj_x, obj_y, obj_x + width, obj_y + height], (x_seg, y_seg)


def initial_frame():
    objects = []
    used_segments = []
    octo_rot = random.choice(ROTATIONS)
    box, segment = place_obj(OCTOPUS, octo_rot, used_segments)
    used_segments.append(segment)

    octo_obj = {
        "position": box,
        "colour": "red",
        "class": "octopus",
        "rotation": octo_rot
    }
    # objects.append(octo_obj)

    num_fish = random.randint(MIN_OBJ, MAX_OBJ)
    num_bags = random.randint(MIN_OBJ, MAX_OBJ)
    num_rocks = random.randint(MIN_OBJ, MAX_OBJ)

    for fish in range(num_fish):
        rot = random.choice(ROTATIONS)
        box, segment = place_obj(FISH, rot, used_segments)
        used_segments.append(segment)

        obj = {
            "position": box,
            "colour": "silver",
            "class": "fish",
            "rotation": rot
        }
        objects.append(obj)

    for bag in range(num_bag):
        rot = random.choice(ROTATIONS)
        box, segment = place_obj(BAG, rot, used_segments)
        used_segments.append(segment)

        obj = {
            "position": box,
            "colour": "white",
            "class": "bag",
            "rotation": rot
        }
        objects.append(obj)

    for rock in range(num_rocks):
        box, segment = place_obj(ROCK, 0, used_segments)
        used_segments.append(segment)

        colour = random.choice(ROCK_COLOURS)
        obj = {
            "position": box,
            "colour": colour,
            "class": "rock",
            "rotation": rot
        }
        objects.append(obj)

    return objects, octo_obj
