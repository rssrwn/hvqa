import numpy as np

from hvqa.dataset.definitions import *
from hvqa.util.exceptions import *


class Drawer:

    @staticmethod
    def draw_frame(frame_dict):
        """
        Draw a frame from a dictionary description of the frame

        :param frame_dict: Dictionary corresponding to frame to draw
        :return: Numpy array (RGB) of image
        """

        red = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8) * BACKGROUND_R
        green = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8) * BACKGROUND_G
        blue = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8) * BACKGROUND_B
        img = np.stack([red, green, blue], axis=2)

        # Draw objects
        for obj in frame_dict["objects"]:
            obj_type = obj["class"]
            if obj_type == "octopus":
                Drawer._draw_octopus(img, obj)
            elif obj_type == "fish":
                Drawer._draw_fish(img, obj)
            elif obj_type == "bag":
                Drawer._draw_bag(img, obj)
            elif obj_type == "rock":
                Drawer._draw_rock(img, obj)
            else:
                raise UnknownObjectTypeException()

        return img

    @staticmethod
    def _draw_octopus(img, octopus):
        x1, y1, x2, y2 = octopus["position"]
        rotation = octopus["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        octo_rgb = Drawer._get_obj_colour(octopus)

        # Body
        for i in range(x1 + 6, x1 + 11):
            Drawer._draw_pixel(img, i, y1, octo_rgb, rotation, x_centre, y_centre)

        for i in range(x1 + 5, x1 + 12):
            Drawer._draw_pixel(img, i, y1 + 1, octo_rgb, rotation, x_centre, y_centre)

        for i in range(x1 + 4, x1 + 13):
            for j in range(y1 + 2, y1 + 12):
                Drawer._draw_pixel(img, i, j, octo_rgb, rotation, x_centre, y_centre)

        # Outer three arms on either side
        for i in range(4):
            Drawer._draw_pixel(img, x1 + i, y1 + 12 - i, octo_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 16 - i, y1 + 12 - i, octo_rgb, rotation, x_centre, y_centre)

            Drawer._draw_pixel(img, x1 + i, y1 + 15 - i, octo_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 16 - i, y1 + 15 - i, octo_rgb, rotation, x_centre, y_centre)

            Drawer._draw_pixel(img, x1 + 2 + i, y1 + 16 - i, octo_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 14 - i, y1 + 16 - i, octo_rgb, rotation, x_centre, y_centre)

        # Inner arms
        for i in range(3):
            Drawer._draw_pixel(img, x1 + 5 + i, y1 + 16 - i, octo_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 11 - i, y1 + 16 - i, octo_rgb, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 3, y1 + 10, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 13, y1 + 10, octo_rgb, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 5, y1 + 12, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 11, y1 + 12, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 8, y1 + 12, octo_rgb, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 7, y1 + 13, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 9, y1 + 13, octo_rgb, rotation, x_centre, y_centre)

        # Eyes
        Drawer._draw_pixel(img, x1 + 6, y1 + 4, BLACK_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 7, y1 + 4, BLACK_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 9, y1 + 4, BLACK_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 10, y1 + 4, BLACK_RGB, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_fish(img, fish):
        x1, y1, x2, y2 = fish["position"]
        rotation = fish["rotation"]
        x_centre, y_centre = Drawer._update_centres((x1, y1, x2, y2), rotation)

        # Body
        for i in range(x1 + 2, x1 + 7):
            for j in range(y1 + 1, y1 + 8):
                Drawer._draw_pixel(img, i, j, FISH_RGB, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 4, y1, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1, y1 + 5, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 1, y1 + 4, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 7, y1 + 4, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 8, y1 + 5, FISH_RGB, rotation, x_centre, y_centre)

        # Tail
        Drawer._draw_pixel(img, x1 + 4, y1 + 8, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 3, y1 + 9, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 2, y1 + 10, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 5, y1 + 9, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 6, y1 + 10, FISH_RGB, rotation, x_centre, y_centre)

        # Eyes
        Drawer._draw_pixel(img, x1 + 4, y1 + 2, BLACK_RGB, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_bag(img, bag):
        x1, y1, x2, y2 = bag["position"]
        rotation = bag["rotation"]
        x_centre, y_centre = Drawer._update_centres((x1, y1, x2, y2), rotation)

        # Body
        for i in range(x1 + 1, x1 + 10):
            for j in range(y1 + 6, y1 + 14):
                Drawer._draw_pixel(img, i, j, BAG_RGB, rotation, x_centre, y_centre)

        # Handles
        for i in range(5):
            Drawer._draw_pixel(img, x1 + 1, y1 + 1 + i, BAG_RGB, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 9, y1 + 1 + i, BAG_RGB, rotation, x_centre, y_centre)

        # Grey outline
        for i in range(15):
            Drawer._draw_pixel(img, x1, y1 + i, GREY_RBG, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 10, y1 + i, GREY_RBG, rotation, x_centre, y_centre)

        for i in range(11):
            Drawer._draw_pixel(img, x1 + i, y1 + 14, GREY_RBG, rotation, x_centre, y_centre)

        for i in range(7):
            Drawer._draw_pixel(img, x1 + 2 + i, y1 + 5, GREY_RBG, rotation, x_centre, y_centre)

        for i in range(6):
            Drawer._draw_pixel(img, x1 + 2, y1 + i, GREY_RBG, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 8, y1 + i, GREY_RBG, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 1, y1, GREY_RBG, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 9, y1, GREY_RBG, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_rock(img, rock):
        x1, y1, x2, y2 = rock["position"]
        rotation = rock["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        rock_rgb = Drawer._get_obj_colour(rock)

        # Diagonals
        lengths = [6, 7, 8, 9, 8, 9, 8, 7, 6, 5, 4, 3]
        starts = [(x1 + 1, y1 + 6), (x1 + 1, y1 + 7), (x1 + 1, y1 + 8), (x1 + 1, y1 + 9), (x1 + 2, y1 + 9),
                  (x1 + 2, y1 + 10), (x1 + 3, y1 + 10), (x1 + 4, y1 + 10), (x1 + 5, y1 + 10), (x1 + 6, y1 + 10),
                  (x1 + 7, y1 + 10), (x1 + 8, y1 + 10)]

        for length, (start_x, start_y) in zip(lengths, starts):
            for i in range(length):
                Drawer._draw_pixel(img, start_x + i, start_y - i, rock_rgb, rotation, x_centre, y_centre)

        # Grey spots
        for i in range(2):
            for j in range(2):
                Drawer._draw_pixel(img, x1 + 2 + i, y1 + 7 + j, GREY_RBG, rotation, x_centre, y_centre)
                Drawer._draw_pixel(img, x1 + 3 + i, y1 + 6 + j, GREY_RBG, rotation, x_centre, y_centre)

                Drawer._draw_pixel(img, x1 + 7 + i, y1 + 2 + j, GREY_RBG, rotation, x_centre, y_centre)
                Drawer._draw_pixel(img, x1 + 6 + i, y1 + 3 + j, GREY_RBG, rotation, x_centre, y_centre)

                Drawer._draw_pixel(img, x1 + 7 + i, y1 + 7 + j, GREY_RBG, rotation, x_centre, y_centre)
                Drawer._draw_pixel(img, x1 + 6 + i, y1 + 8 + j, GREY_RBG, rotation, x_centre, y_centre)
                Drawer._draw_pixel(img, x1 + 8 + i, y1 + 6 + j, GREY_RBG, rotation, x_centre, y_centre)

        # Grey outline
        for i in range(7):
            Drawer._draw_pixel(img, x1 + i, y1 + 6 - i, GREY_RBG, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 2 + i, y1 + 11, GREY_RBG, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 11, y1 + 2 + i, GREY_RBG, rotation, x_centre, y_centre)

        for i in range(4):
            Drawer._draw_pixel(img, x1, y1 + 6 + i, GREY_RBG, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 6 + i, y1, GREY_RBG, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1 + 1, y1 + 10, GREY_RBG, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 10, y1 + 1, GREY_RBG, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 9, y1 + 10, GREY_RBG, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 10, y1 + 9, GREY_RBG, rotation, x_centre, y_centre)

    @staticmethod
    def _update_centres(pos, rot):
        """
        Update centres for non-square objects

        :param pos: position
        :param rot: Rotation
        :return: x_centre, y_centre
        """

        x1, y1, x2, y2 = pos
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)
        w_h_diff = abs((x2 - x1) - (y2 - y1))
        if rot == 1:
            y_centre += (w_h_diff // 2)
        elif rot == 3:
            x_centre -= (w_h_diff // 2)

        return x_centre, y_centre

    @staticmethod
    def _get_obj_colour(obj):
        colour = obj["colour"]
        if colour == "brown":
            rgb = BROWN_ROCK_RGB
        elif colour == "blue":
            rgb = BLUE_ROCK_RGB
        elif colour == "purple":
            rgb = PURPLE_ROCK_RGB
        elif colour == "green":
            rgb = GREEN_ROCK_RGB
        elif colour == "red":
            rgb = OCTO_RGB
        else:
            raise UnknownPropertyException(f"Unknown rock colour: {colour}")

        return rgb

    @staticmethod
    def _set_pixel_colour(img, x, y, rgb_tuple):
        r, g, b = rgb_tuple
        img[y, x, 0] = r
        img[y, x, 1] = g
        img[y, x, 2] = b

    @staticmethod
    def _draw_pixel(img, x, y, rgb_tuple, rotation, x_centre, y_centre):
        x_diff = x_centre - x
        y_diff = y_centre - y

        if rotation == 0:
            Drawer._set_pixel_colour(img, x, y, rgb_tuple)
        elif rotation == 1:
            Drawer._set_pixel_colour(img, x_centre + y_diff, y_centre - x_diff, rgb_tuple)  # +1
        elif rotation == 2:
            Drawer._set_pixel_colour(img, x_centre + x_diff, y_centre + y_diff, rgb_tuple)
        elif rotation == 3:
            Drawer._set_pixel_colour(img, x_centre - y_diff, y_centre + x_diff, rgb_tuple)  # -1
