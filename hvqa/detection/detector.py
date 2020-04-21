import torch
import torchvision.transforms as T

from hvqa.util.video_repr import Obj, Frame, Video
from hvqa.util.definitions import CLASSES
from hvqa.util.func import get_device


_transform = T.Compose([
    T.ToTensor(),
])


class _AbsDetector:
    def detect_objs(self, frames):
        """
        Detect all objects in a video

        :param frames: List of PIL frames
        :return: Video object containing info on objects in video
        """

        raise NotImplementedError


class NeuralDetector(_AbsDetector):
    def __init__(self, model):
        super(NeuralDetector, self).__init__()

        self.model = model

    def detect_objs(self, frames):
        imgs_trans = [_transform(img) for img in frames]
        imgs_batch = torch.stack(imgs_trans)

        device = get_device()
        imgs_batch = imgs_batch.to(device)

        with torch.no_grad():
            detector_out = self.model(imgs_batch)

        frames_objs = []
        for idx, frame in enumerate(detector_out):
            img = frames[idx]
            bboxs = [bbox.numpy() for bbox in list(frame["boxes"])]
            bboxs = [tuple(map(round, bbox)) for bbox in bboxs]
            labels = [CLASSES[label.numpy() - 1] for label in list(frame["labels"])]

            objs = []
            for obj_idx, bbox in enumerate(bboxs):
                label = labels[obj_idx]
                obj = Obj(label, bbox)
                obj.set_image(img)
                objs.append(obj)

            frame = Frame(objs)
            frames_objs.append(frame)

        video = Video(frames_objs)
        return video
