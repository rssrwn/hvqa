import random

from dataset.creation.frame_object import FrameObject


FRAME_SIZE = 128
NUM_SEGMENTS = 4
SEGMENT_SIZE = FRAME_SIZE / NUM_SEGMENTS
EDGE = 3

MIN_OBJ = 2
MAX_OBJ = 4


class Frame:

    def __init__(self):
        self.static_objects = []
        self.remaining_segments = [(i, j) for i in range(NUM_SEGMENTS) for j in range(NUM_SEGMENTS)]
        self.octopus = None

    def obj_box(self, obj_size, rotation):
        x_seg, y_seg = random.choice(self.remaining_segments)
        self.remaining_segments.remove((x_seg, y_seg))

        width = obj_size[0]
        height = obj_size[1]
        if rotation == 1 or rotation == 3:
            width = obj_size[1]
            height = obj_size[0]

        obj_x = random.randint((x_seg * SEGMENT_SIZE) + EDGE, ((x_seg + 1) * SEGMENT_SIZE) - (width + EDGE))
        obj_y = random.randint((y_seg * SEGMENT_SIZE) + EDGE, ((y_seg + 1) * SEGMENT_SIZE) - (height + EDGE))

        return [obj_x, obj_y, obj_x + width, obj_y + height]

    def random_frame(self):
        octo = FrameObject(self).random_obj("octopus")
        self.octopus = octo

        num_fish = random.randint(MIN_OBJ, MAX_OBJ)
        num_bags = random.randint(MIN_OBJ, MAX_OBJ)
        num_rocks = random.randint(MIN_OBJ, MAX_OBJ)

        for _ in range(num_fish):
            fish = FrameObject(self).random_obj("fish")
            self.static_objects.append(fish)

        for _ in range(num_bags):
            bag = FrameObject(self).random_obj("bag")
            self.static_objects.append(bag)

        for _ in range(num_rocks):
            rock = FrameObject(self).random_obj("rock")
            self.static_objects.append(rock)

    def to_dict(self):
        statics = [obj.to_dict() for obj in self.static_objects]
        return {
            "objects": statics + self.octopus
        }
