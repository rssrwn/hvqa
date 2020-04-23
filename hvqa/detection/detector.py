import torch
import torchvision.transforms as T

from hvqa.detection.models import DetectionBackbone, DetectionModel
from hvqa.util.environment import Obj, Frame, Video
from hvqa.util.definitions import CLASSES
from hvqa.util.func import get_device, load_model, save_model
from hvqa.util.interfaces import Detector


_transform = T.Compose([
    T.ToTensor(),
])


class NeuralDetector(Detector):
    def __init__(self, model):
        super(NeuralDetector, self).__init__()

        self.model = model

    def train(self, data):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        backbone = DetectionBackbone()
        detector_model = DetectionModel(backbone)
        detector_model.eval()
        detector = NeuralDetector(detector_model)
        return detector

    @staticmethod
    def load(path):
        backbone = DetectionBackbone()
        detector_model = load_model(DetectionModel, path, backbone)
        detector_model.eval()
        detector = NeuralDetector(detector_model)
        return detector

    def save(self, path):
        save_model(self.model, path)

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
            bboxs = [bbox.cpu().numpy() for bbox in list(frame["boxes"])]
            bboxs = [tuple(map(round, bbox)) for bbox in bboxs]
            labels = [CLASSES[label.cpu().numpy() - 1] for label in list(frame["labels"])]

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
