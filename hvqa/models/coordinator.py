from PIL import ImageDraw
from PIL import ImageFont


class Coordinator:
    """
    Coordination class which will be able to answer questions on videos
    """

    def __init__(self, detector, prop_classifier, tracker):
        self.detector = detector
        self.prop_classifier = prop_classifier
        self.tracker = tracker

        self._img_size = 256
        self._visualise_mult = 4

        self.font = ImageFont.truetype("./lib/fonts/Arial Unicode.ttf", size=18)

    def analyse_video(self, frames):
        """
        Analyse a video.
        For now this method will:
        1. detect objects
        2. Extract logical properties from object
        3. Visualise each frame

        :param frames: List of PIL Images
        """

        # Batch all frames in a video
        video = self._extract_objs(frames)

        # Batch all objects in each frame
        for frame in video.frames:
            objs = frame.objs
            self._extract_props_(objs)
            ids = self.tracker.process_frame(objs)
            self._add_ids(objs, ids)

        self.tracker.reset()
        new_frames = self._add_info_to_video(frames, video)
        self.visualise(new_frames)

    @staticmethod
    def visualise(imgs):
        for img in imgs:
            img.show()

    @staticmethod
    def _add_ids(objs, ids):
        for idx, obj in enumerate(objs):
            obj.id = ids[idx]

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

    def _extract_objs(self, frames):
        """
        Builds structured knowledge from video frames
        Extracts bboxs and class for each object in each frame

        :param frames: List of PIL frames
        :return: Video object for video
        """

        video = self.detector.detect_objs(frames)
        return video

    def _extract_props_(self, objs):
        """
        Extract properties from each object and add them to the object
        Note: Modifies objects in-place with new properties

        :param objs: List of Objs
        """

        obj_imgs = [obj.img for obj in objs]
        props = self.prop_classifier.extract_props(obj_imgs)
        for idx, (colour, rot, cls) in enumerate(props):
            obj = objs[idx]
            obj.colour = colour
            obj.rot = rot

            # We use class from detector (if this line is commented)
            # obj.cls = cls

    # def _extract_relations_(self, frame):
    #     """
    #     Extract relations between objects in a frame
    #     Note: Modifies frame in-place to add relations
    #
    #     :param frame: Frame obj
    #     """
    #
    #     pass
