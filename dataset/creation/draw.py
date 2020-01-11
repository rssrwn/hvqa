import numpy as np
import cv2

from dataset.creation.definitions import *


class Drawer:

    @staticmethod
    def draw_frame(frame):
        size = frame.frame_size

        # Create background (opencv uses BGR)
        red = np.ones((size, size)) * BACKGROUND_B
        green = np.ones((size, size)) * BACKGROUND_G
        blue = np.ones((size, size)) * BACKGROUND_R
        img = np.stack([red, green, blue], axis=2)

        Drawer._draw_octopus(img, frame.octopus)

        for obj in frame.static_objects:
            if obj.obj_type == "fish":
                Drawer._draw_fish(img, obj)
            elif obj.obj_type == "bag":
                Drawer._draw_bag(img, obj)
            elif obj.obj_type == "rock":
                Drawer._draw_rock(img, obj)
            else:
                raise UnknownObjectType()

        return img

    @staticmethod
    def _draw_octopus(img, octopus):
        x1, y1, x2, y2 = octopus.position

        # Body
        for i in range(x1+4, x1+7):
            Drawer._set_pixel_colour(img, i, y1+1, OCTO_RGB)

        for i in range(x1+3, x1+8):
            for j in range(y1+2, y1+7):
                Drawer._set_pixel_colour(img, i, j, OCTO_RGB)

        # Arms
        for i in range(x1+1, x1+10):
            Drawer._set_pixel_colour(img, i, y1+5, OCTO_RGB)

        Drawer._set_pixel_colour(img, x1, y1+6, OCTO_RGB)
        Drawer._set_pixel_colour(img, x2, y1+6, OCTO_RGB)
        Drawer._set_pixel_colour(img, x1+2, y1+7, OCTO_RGB)
        Drawer._set_pixel_colour(img, x1+1, y1+8, OCTO_RGB)
        Drawer._set_pixel_colour(img, x2-2, y2-2, OCTO_RGB)
        Drawer._set_pixel_colour(img, x1+5, y2-2, OCTO_RGB)
        Drawer._set_pixel_colour(img, x1+4, y2-1, OCTO_RGB)
        Drawer._set_pixel_colour(img, x1+6, y2-1, OCTO_RGB)

        # Eyes
        Drawer._set_pixel_colour(img, x1+4, y1+3, BLACK_RGB)
        Drawer._set_pixel_colour(img, x1+6, y1+3, BLACK_RGB)

    @staticmethod
    def _draw_fish(img, fish):
        x1, y1, x2, y2 = fish.position

        # Body
        for i in range(x1+2, x1+7):
            for j in range(y1+1, y1+4):
                Drawer._set_pixel_colour(img, i, j, FISH_RGB)

        Drawer._set_pixel_colour(img, x1+4, y1, FISH_RGB)
        Drawer._set_pixel_colour(img, x1+4, y2, FISH_RGB)

        # Tail
        Drawer._set_pixel_colour(img, x1, y1+1, FISH_RGB)
        Drawer._set_pixel_colour(img, x1, y1+3, FISH_RGB)
        Drawer._set_pixel_colour(img, x1+1, y1+2, FISH_RGB)

        # Eyes
        Drawer._set_pixel_colour(img, x1+5, y1+2, BLACK_RGB)

    @staticmethod
    def _draw_bag(img, bag):
        x1, y1, x2, y2 = bag.position

        # Body
        for i in range(x1, x2):
            for j in range(y1+2, y2):
                Drawer._set_pixel_colour(img, i, j, BAG_RGB)

        # Handles
        Drawer._set_pixel_colour(img, x1, y1, BAG_RGB)
        Drawer._set_pixel_colour(img, x1, y1+1, BAG_RGB)
        Drawer._set_pixel_colour(img, x2, y1, BAG_RGB)
        Drawer._set_pixel_colour(img, x2, y1+1, BAG_RGB)

    @staticmethod
    def _draw_rock(img, rock):
        x1, y1, x2, y2 = rock.position
        colour = rock.colour
        if colour == "brown":
            rock_rgb = BROWN_ROCK_RGB
        elif colour == "blue":
            rock_rgb = BLUE_ROCK_RGB
        elif colour == "purple":
            rock_rgb = PURPLE_ROCK_RGB
        elif colour == "green":
            rock_rgb = GREEN_ROCK_RGB
        else:
            raise UnknownPropertyException(f"Unknown rock colour: {colour}")

        # Main body
        for i in range(x1+1, x2):
            for j in range(y1+1, y2):
                Drawer._set_pixel_colour(img, i, j, rock_rgb)

        for i in range(x1+2, x1+5):
            Drawer._set_pixel_colour(img, i, y1, rock_rgb)
            Drawer._set_pixel_colour(img, i, y2, rock_rgb)

        for j in range(y1+2, y1+4):
            Drawer._set_pixel_colour(img, x1, j, rock_rgb)
            Drawer._set_pixel_colour(img, x2, j, rock_rgb)

        # Black section
        for i in range(x1+3, x1+5):
            for j in range(y1+3, y1+5):
                Drawer._set_pixel_colour(img, i, j, BLACK_RGB)
                Drawer._set_pixel_colour(img, i+1, j-1, BLACK_RGB)

    @staticmethod
    def _set_pixel_colour(img, x, y, rgb_tuple):
        r, g, b = rgb_tuple
        img[x, y, 0] = b
        img[x, y, 1] = g
        img[x, y, 2] = r
