import torch
import torch.nn as nn
import torchvision.transforms as T

from hvqa.detection.models import DetectionBackbone, DetectionModel
from hvqa.spec.repr import Obj, Frame
from hvqa.spec.definitions import CLASSES
from hvqa.util.func import get_device, load_model, save_model
from hvqa.util.interfaces import Detector, Trainable


_transform = T.Compose([
    T.ToTensor(),
])


class NeuralDetector(Detector, Trainable):
    def __init__(self, spec, model):
        super(NeuralDetector, self).__init__()

        gpus = torch.cuda.device_count()
        model = nn.DataParallel(model) if gpus > 1 else model
        print(f"Object detector using {gpus} cuda devices.")

        self.spec = spec
        self._device = get_device()
        self.model = model.to(self._device)

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        backbone = DetectionBackbone()
        detector_model = DetectionModel(backbone)
        detector_model.eval()
        detector = NeuralDetector(spec, detector_model)
        return detector

    @staticmethod
    def load(spec, path):
        backbone = DetectionBackbone()
        detector_model = load_model(DetectionModel, path, backbone)
        detector_model.eval()
        detector = NeuralDetector(spec, detector_model)
        return detector

    def save(self, path):
        save_model(self.model, path)

    def detect_objs(self, frames):
        """
        Detect objects in the images

        :param frames: List of PIL images
        :return: List of Frame objs
        """

        self.model.eval()

        imgs_trans = [_transform(img) for img in frames]
        imgs_batch = torch.stack(imgs_trans)
        imgs_batch = imgs_batch.to(self._device)

        with torch.no_grad():
            detector_out = self.model(imgs_batch)

        frames_objs = []
        for idx, frame in enumerate(detector_out):
            img = frames[idx]
            bboxs = [bbox.cpu().numpy() for bbox in list(frame["boxes"])]
            bboxs = [tuple(map(round, bbox)) for bbox in bboxs]

            # TODO convert to use spec
            labels = [CLASSES[label.cpu().numpy() - 1] for label in list(frame["labels"])]

            objs = []
            for obj_idx, bbox in enumerate(bboxs):
                label = labels[obj_idx]
                obj = Obj(self.spec, label, bbox)
                obj.set_image(img)
                objs.append(obj)

            frame = Frame(self.spec, objs)
            frames_objs.append(frame)

        return frames_objs
