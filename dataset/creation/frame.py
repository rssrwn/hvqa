import random

from dataset.creation.frame_object import FrameObject
from dataset.creation.definitions import *


class Frame:

    def __init__(self):
        """
        Initialisation method
        """

        self.static_objects = []
        self._remaining_segments = [(i, j) for i in range(NUM_SEGMENTS) for j in range(NUM_SEGMENTS)]
        self.octopus = None
        self.frame_size = FRAME_SIZE

    def obj_box(self, obj_size, rotation):
        """
        Create the bounding box for an object

        :param obj_size: Object's width and height when upright
        :param rotation: Rotation (0: up, 1: left, 2: down, 3: right)
        :return: List of four points: x1, y1, x2, y2
        """

        x_seg, y_seg = random.choice(self._remaining_segments)
        self._remaining_segments.remove((x_seg, y_seg))

        width = obj_size[0]
        height = obj_size[1]
        if rotation == 1 or rotation == 3:
            width = obj_size[1]
            height = obj_size[0]

        obj_x = random.randint((x_seg * SEGMENT_SIZE) + EDGE, ((x_seg + 1) * SEGMENT_SIZE) - (width + EDGE))
        obj_y = random.randint((y_seg * SEGMENT_SIZE) + EDGE, ((y_seg + 1) * SEGMENT_SIZE) - (height + EDGE))

        return [obj_x, obj_y, obj_x + width, obj_y + height]

    def random_frame(self):
        """
        Create a randomly initialised frame
        """

        octo = FrameObject(self)
        octo.random_obj("octopus")
        self.octopus = octo
        self._gen_static_objects("fish")
        self._gen_static_objects("bag")
        self._gen_static_objects("rock")

    def _gen_static_objects(self, obj_type):
        num_objs = random.randint(MIN_OBJ, MAX_OBJ)
        for _ in range(num_objs):
            obj = FrameObject(self)
            obj.random_obj(obj_type)
            self.static_objects.append(obj)

    def move(self):
        """
        Move or rotate the octopus

        :return Next frame, with all objects updated and list of events which occurred
        """

        next_frame = Frame()
        next_frame.static_objects = [obj.copy(next_frame) for obj in self.static_objects]

        # If the octopus has already disappeared then nothing happens
        if self.octopus is None:
            next_frame.octopus = None
            return next_frame, ["no event"]

        next_frame.octopus = self.octopus.copy(next_frame)

        rand = random.random()
        if rand <= ROT_PROB:
            next_frame.octopus.rotate()
            event = "rotate"
        else:
            event = next_frame.octopus.move(MOVE_PIXELS, FRAME_SIZE)

        update_events = next_frame.update_frame()
        return next_frame, update_events + [event]

    def update_frame(self):
        """
        Update the frame to account for the object getting close to an object
        If the octopus is close to a fish, the fish disappears
        If the octopus is close to a bag, both objects disappear
        If the octopus is close to a rock, the octopus changes colour to the rock's colour

        :return: List of events ('disappear' or 'colour change')
        """

        events = []
        remove_octopus = False
        for obj in self.static_objects:
            if self.close_to_octopus(obj):
                if obj.obj_type == "fish":
                    self.static_objects.remove(obj)
                    events.append('disappear')

                elif obj.obj_type == "bag":
                    self.static_objects.remove(obj)
                    remove_octopus = True
                    events.append('disappear')

                elif obj.obj_type == "rock":
                    self.octopus.colour = obj.colour
                    events.append('colour change')

                else:
                    raise UnknownObjectType()

        if remove_octopus:
            self.octopus = None

        return events

    def close_to_octopus(self, obj):
        """
        Returns whether the object is close to the octopus
        A border is created around the octopus, an object is close if it is within the border

        :param obj: Object to check
        :return: bool
        """

        octo_x1, octo_y1, octo_x2, octo_y2 = self.octopus.position
        octo_x1 -= CLOSE_OCTO
        octo_x2 += CLOSE_OCTO
        octo_y1 -= CLOSE_OCTO
        octo_y2 += CLOSE_OCTO

        x1, y1, x2, y2 = obj.position
        obj_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        for x, y in obj_corners:
            if octo_x1 <= x <= octo_x2 and octo_y1 <= y <= octo_y2:
                return True

        return False

    def to_dict(self):
        statics = [obj.to_dict() for obj in self.static_objects]
        octopus = None
        if self.octopus is not None:
            octopus = self.octopus.to_dict()

        return {
            "objects": statics + [octopus]
        }
