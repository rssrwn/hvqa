

class UnknownObjectType(BaseException):
    pass


class UnknownPropertyException(BaseException):
    pass


ROTATIONS = [0, 1, 2, 3]
ROCK_COLOURS = ["brown", "blue", "purple", "green"]
OCTO_COLOUR = "red"
FISH_COLOUR = "silver"
BAG_COLOUR = "white"

OCTOPUS = (11, 11)
FISH = (5, 7)
BAG = (5, 7)
ROCK = (7, 7)

NUM_FRAMES = 32

FRAME_SIZE = 128
NUM_SEGMENTS = 4
SEGMENT_SIZE = FRAME_SIZE / NUM_SEGMENTS
EDGE = 3
CLOSE_OCTO = 5

MIN_OBJ = 2
MAX_OBJ = 4

ROT_PROB = 0.33
MOVE_PIXELS = 12

# *** Colour definitions ***

BLACK_RGB = (0, 0, 0)

# Background
BACKGROUND_R = 62
BACKGROUND_G = 193
BACKGROUND_B = 179

# Objects
OCTO_RGB = (226, 29, 98)
FISH_RGB = (192, 190, 188)
BAG_RGB = (255, 255, 255)

BROWN_ROCK_RGB = (182, 122, 28)
BLUE_ROCK_RGB = (0, 0, 255)
PURPLE_ROCK_RGB = (182, 37, 218)
GREEN_ROCK_RGB = (0, 255, 0)
