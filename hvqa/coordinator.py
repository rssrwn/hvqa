import torch
from PIL import ImageDraw
from PIL import ImageFont

from hvqa.util.definitions import COLOURS, ROTATIONS, CLASSES
import hvqa.util.util as util


class Coordinator:
    """
    Coordination class which will be able to answer questions on videos
    """

    def __init__(self, detector, detector_transform, prop_classifier, prop_transform, tracker):
        self.detector = detector
        self.detector_transform = detector_transform
        self.prop_classifier = prop_classifier
        self.prop_transform = prop_transform
        self.tracker = tracker

        self._img_size = 256
        self._visualise_mult = 4

        self.font = ImageFont.truetype("./lib/fonts/Arial Unicode.ttf", size=18)

    def analyse_video(self, video):
        """
        Analyse a video.
        For now this method will:
        1. detect objects
        2. Extract logical properties from object
        3. Visualise each frame

        :param video: List of PIL Images
        """

        # Pass all images in a batch through detection model
        imgs = self._extract_objs(video)

        # Pass objects in each image as a batch through property network
        imgs_objs = []
        for img, bboxs, labels in imgs:
            bboxs = [bbox.numpy() for bbox in list(bboxs)]
            labels = [label.numpy() for label in list(labels)]
            objs = self._extract_props(img, bboxs, labels)
            ids = self.tracker.process_frame(objs)
            self._add_ids(objs, ids)
            imgs_objs.append(objs)

        new_imgs = [self._add_info_to_img(video[idx], info) for idx, info in enumerate(imgs_objs)]
        self.visualise(new_imgs)

    def visualise(self, imgs):
        for img in imgs:
            img.show()

    @staticmethod
    def _add_ids(objs, ids):
        for idx, obj in enumerate(objs):
            obj["id"] = ids[idx]

    def _add_info_to_img(self, img, img_info):
        """
        Increase size of image, draw bboxs and write object info

        :param img: PIL Image
        :param img_info: List of dictionaries each with object info
        :return: PIL Image with info
        """

        new_size = self._img_size * self._visualise_mult
        img = img.resize((new_size, new_size))

        draw = ImageDraw.Draw(img)

        for obj_info in img_info:
            bbox = tuple(map(lambda coord: coord * self._visualise_mult, obj_info["position"]))
            colour = obj_info["colour"]
            rotation = obj_info["rotation"]
            label = obj_info["class"]
            id = obj_info["id"]

            x1, y1, x2, y2 = bbox

            self._add_bbox(draw, bbox, colour)
            draw.text((x1, y1 - 35), label, fill=colour, font=self.font)
            draw.text((x2 - 20, y2 + 10), "Rot: " + str(rotation), fill=colour, font=self.font)
            draw.text((x1 - 30, y2 + 10), "Id: " + str(id), fill=colour, font=self.font)

        return img

    def _add_bbox(self, drawer, position, colour):
        x1, y1, x2, y2 = position
        x1 = round(x1) - 1
        y1 = round(y1) - 1
        x2 = round(x2) + (1 * self._visualise_mult)
        y2 = round(y2) + (1 * self._visualise_mult)
        drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour, width=3)

    def _extract_objs(self, imgs):
        imgs_trans = [self.detector_transform(img) for img in imgs]
        imgs_batch = torch.stack(imgs_trans)

        with torch.no_grad():
            detector_out = self.detector(imgs_batch)

        detection = [(imgs[idx], img["boxes"], img["labels"]) for idx, img in enumerate(detector_out)]
        return detection

    def _extract_props(self, img, bboxs, labels):
        objs_img = [util.collect_obj(img, bbox) for bbox in bboxs]
        objs_img_trans = [self.prop_transform(obj) for obj in objs_img]
        objs_batch = torch.stack(objs_img_trans)

        with torch.no_grad():
            preds = self.prop_classifier(objs_batch)

        preds = [torch.max(pred, dim=1)[1].numpy() for pred in preds]
        colours = [COLOURS[idx] for idx in preds[0]]
        rotations = [ROTATIONS[idx] for idx in preds[1]]
        classes = [CLASSES[idx] for idx in preds[2]]

        objs = []

        # Note: we use labels from the property network, not from the detector
        for idx, bbox in enumerate(bboxs):
            obj = {
                "image": objs_img[idx],
                "position": tuple(map(round, bbox)),
                "colour": colours[idx],
                "rotation": rotations[idx],
                "class": classes[idx]
            }
            objs.append(obj)

        return objs
