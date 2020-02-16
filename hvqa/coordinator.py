import torch

import hvqa.util as util


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

    def analyse_video(self, video):
        """
        Analyse a video.
        For now this method will:
        1. detect objects
        2. Extract logical properties from object
        3. Visualise each frame

        :param video: List of PIL Images
        """

        for img in video:
            bboxs, labels = self._extract_objs(img)
            # for idx, bbox in enumerate(bboxs):
            #     obj = self._extract_props(img, bbox)

    def _extract_objs(self, img):
        img = self.detector_transform(img)
        detector_out = self.detector(img)
        print(detector_out)
        bboxs = detector_out["boxes"]
        labels = detector_out["labels"]
        return bboxs, labels

    def _extract_props(self, img, bbox):
        obj_img = util.collect_obj(img, bbox)
        obj_img = self.prop_transform(obj_img)
        props = self.prop_classifier(obj_img)
