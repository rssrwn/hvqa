import numpy as np

from definitions import *


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
            print(obj)
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
                raise UnknownObjectType()

        return img

    @staticmethod
    def _draw_octopus(img, octopus):
        x1, y1, x2, y2 = octopus["position"]
        rotation = octopus["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        octo_rgb = Drawer._get_obj_colour(octopus)

        # Body
        for i in range(x1 + 4, x1 + 7):
            Drawer._draw_pixel(img, i, y1 + 1, octo_rgb, rotation, x_centre, y_centre)

        for i in range(x1 + 3, x1 + 8):
            for j in range(y1 + 2, y1 + 7):
                Drawer._draw_pixel(img, i, j, octo_rgb, rotation, x_centre, y_centre)

        # Arms
        for i in range(x1 + 1, x1 + 10):
            Drawer._draw_pixel(img, i, y1 + 5, octo_rgb, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1, y1 + 6, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 10, y1 + 6, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 2, y1 + 7, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 1, y1 + 8, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 9, y1 + 8, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 8, y1 + 7, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 5, y1 + 7, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 4, y1 + 8, octo_rgb, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 6, y1 + 8, octo_rgb, rotation, x_centre, y_centre)

        # Eyes
        Drawer._draw_pixel(img, x1 + 4, y1 + 3, BLACK_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 6, y1 + 3, BLACK_RGB, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_fish(img, fish):
        x1, y1, x2, y2 = fish["position"]
        rotation = fish["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        # Body
        for i in range(x1 + 1, x1 + 4):
            for j in range(y1, y1 + 5):
                Drawer._draw_pixel(img, i, j, FISH_RGB, rotation, x_centre, y_centre)

        Drawer._draw_pixel(img, x1, y1 + 2, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 4, y1 + 2, FISH_RGB, rotation, x_centre, y_centre)

        # Tail
        Drawer._draw_pixel(img, x1 + 1, y1 + 6, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 3, y1 + 6, FISH_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 2, y1 + 5, FISH_RGB, rotation, x_centre, y_centre)

        # Eyes
        Drawer._draw_pixel(img, x1 + 2, y1 + 1, BLACK_RGB, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_bag(img, bag):
        x1, y1, x2, y2 = bag["position"]
        rotation = bag["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        # Body
        for i in range(x1, x1 + 5):
            for j in range(y1 + 2, y1 + 7):
                Drawer._draw_pixel(img, i, j, BAG_RGB, rotation, x_centre, y_centre)

        # Handles
        Drawer._draw_pixel(img, x1, y1, BAG_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1, y1 + 1, BAG_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 4, y1, BAG_RGB, rotation, x_centre, y_centre)
        Drawer._draw_pixel(img, x1 + 4, y1 + 1, BAG_RGB, rotation, x_centre, y_centre)

    @staticmethod
    def _draw_rock(img, rock):
        x1, y1, x2, y2 = rock["position"]
        rotation = rock["rotation"]
        x_centre = x1 + ((x2 - x1) // 2)
        y_centre = y1 + ((y2 - y1) // 2)

        rock_rgb = Drawer._get_obj_colour(rock)

        # Main body
        for i in range(x1 + 1, x1 + 6):
            for j in range(y1 + 1, y1 + 6):
                Drawer._draw_pixel(img, i, j, rock_rgb, rotation, x_centre, y_centre)

        for i in range(x1 + 2, x1 + 5):
            Drawer._draw_pixel(img, i, y1, rock_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, i, y1 + 6, rock_rgb, rotation, x_centre, y_centre)

        for j in range(y1 + 2, y1 + 5):
            Drawer._draw_pixel(img, x1, j, rock_rgb, rotation, x_centre, y_centre)
            Drawer._draw_pixel(img, x1 + 6, j, rock_rgb, rotation, x_centre, y_centre)

        # Black section
        for i in range(x1 + 3, x1 + 5):
            for j in range(y1 + 4, y1 + 6):
                Drawer._draw_pixel(img, i, j, BLACK_RGB, rotation, x_centre, y_centre)
                Drawer._draw_pixel(img, i + 1, j - 1, BLACK_RGB, rotation, x_centre, y_centre)

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
            Drawer._set_pixel_colour(img, x_centre + y_diff, y_centre - x_diff, rgb_tuple)
        elif rotation == 2:
            Drawer._set_pixel_colour(img, x_centre + x_diff, y_centre + y_diff, rgb_tuple)
        elif rotation == 3:
            Drawer._set_pixel_colour(img, x_centre - y_diff, y_centre + x_diff, rgb_tuple)
