import random


ROTATIONS = [0, 1, 2, 3]
ROCK_COLOURS = ["brown", "blue", "purple", "green"]
OCTO_COLOUR = "red"
FISH_COLOUR = "silver"
BAG_COLOUR = "white"

OCTOPUS = (11, 9)
FISH = (7, 5)
BAG = (5, 7)
ROCK = (7, 7)


class UnknownObjectType(BaseException):
    pass


class FrameObject:

    def __init__(self, frame):
        self.frame = frame
        self.obj_type = None
        self.position = None
        self.colour = None
        self.rotation = None

    def to_dict(self):
        return {
            "position": self.position,
            "class": self.obj_type,
            "colour": self.colour,
            "rotation": self.rotation
        }

    def random_obj(self, obj_type):
        if obj_type == "octopus":
            self._init_octopus()
        elif obj_type == "fish":
            self._init_fish()
        elif obj_type == "bag":
            self._init_bag()
        elif obj_type == "rock":
            self._init_rock()
        else:
            raise UnknownObjectType()

    def _init_octopus(self):
        rot = random.choice(ROTATIONS)
        box = self.frame.obj_box(OCTOPUS, rot)
        self.obj_type = "octopus"
        self.position = box
        self.colour = OCTO_COLOUR
        self.rotation = rot

    def _init_fish(self):
        rot = random.choice(ROTATIONS)
        box = self.frame.obj_box(FISH, rot)
        self.obj_type = "fish"
        self.position = box
        self.colour = FISH_COLOUR
        self.rotation = rot

    def _init_bag(self):
        rot = random.choice(ROTATIONS)
        box = self.frame.obj_box(BAG, rot)
        self.obj_type = "bag"
        self.position = box
        self.colour = BAG_COLOUR
        self.rotation = rot

    def _init_rock(self):
        rot = 0
        colour = random.choice(ROCK_COLOURS)
        box = self.frame.obj_box(ROCK, rot)
        self.obj_type = "rock"
        self.position = box
        self.colour = colour
        self.rotation = rot
