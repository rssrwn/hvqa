# Helper classes and definitions


# *** Image and video definitions ***

ROTATIONS = [0, 1, 2, 3]

OCTOPUS = (17, 17)
FISH = (5, 7)
BAG = (5, 7)
ROCK = (7, 7)

NUM_FRAMES = 48

FRAME_SIZE = 128
NUM_SEGMENTS = 4
SEGMENT_SIZE = FRAME_SIZE / NUM_SEGMENTS
EDGE = 3
CLOSE_OCTO = 5

ROT_PROB = 0.1
MOVE_PIXELS = 8


# *** Number of objects ***
MIN_FISH = 2
MAX_FISH = 4

MIN_BAG = 1
MAX_BAG = 2

MIN_ROCK = 2
MAX_ROCK = 4


# *** Colour definitions ***

ROCK_COLOURS = ["brown", "blue", "purple", "green"]
OCTO_COLOUR = "red"
FISH_COLOUR = "silver"
BAG_COLOUR = "white"

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
