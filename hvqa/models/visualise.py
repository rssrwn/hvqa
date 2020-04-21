from PIL import ImageDraw
from PIL import ImageFont


class Visualiser:
    """
    Visualiser class which can produce PIL images with video info
    """

    def __init__(self, model):
        self.model = model

        self._img_size = 256
        self._visualise_mult = 4

        self.font = ImageFont.truetype("./lib/fonts/Arial Unicode.ttf", size=18)

    def visualise(self, frames):
        """
        Visualise a video.
        For now this method will:
        1. detect objects
        2. Extract logical properties from object
        3. Visualise each frame

        :param frames: List of PIL Images
        """

        video = self.model.process(frames)
        new_frames = self._add_info_to_video(frames, video)
        for img in new_frames:
            img.show()

    # TODO: Add relation and event info
    def _add_info_to_video(self, imgs, video):
        """
        For each image in the video:
          - Increase size of image
          - Draw bounding boxes
          - Write object info

        :param imgs: List of PIL Image
        :param video: Frame object corresponding to PIL images
        :return: List of PIL Image with info
        """

        new_size = self._img_size * self._visualise_mult
        imgs = [img.resize((new_size, new_size)) for img in imgs]

        for img_idx, img in enumerate(imgs):
            frame = video.frames[img_idx]
            draw = ImageDraw.Draw(img)

            for obj in frame.objs:
                bbox = tuple(map(lambda coord: coord * self._visualise_mult, obj.pos))
                x1, y1, x2, y2 = bbox
                colour = obj.colour

                self._add_bbox(draw, bbox, colour)
                draw.text((x1, y1 - 35), obj.cls, fill=colour, font=self.font)
                draw.text((x2 - 20, y2 + 10), "Rot: " + str(obj.rot), fill=colour, font=self.font)
                draw.text((x1 - 30, y2 + 10), "Id: " + str(obj.id), fill=colour, font=self.font)

        return imgs

    def _add_bbox(self, drawer, position, colour):
        x1, y1, x2, y2 = position
        x1 = round(x1) - 1
        y1 = round(y1) - 1
        x2 = round(x2) + (1 * self._visualise_mult)
        y2 = round(y2) + (1 * self._visualise_mult)
        drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour, width=3)
