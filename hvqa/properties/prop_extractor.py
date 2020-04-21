import torch
import torchvision.transforms as T

from hvqa.properties.models import PropertyExtractionModel
from hvqa.util.definitions import COLOURS, ROTATIONS, CLASSES
from hvqa.util.func import get_device, load_model, save_model
from hvqa.util.interfaces import Component


_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


class _AbsPropExtractor(Component):
    def extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of tuple [(colour, rotation, class)]
        """

        raise NotImplementedError

    def run_(self, data):
        assert type(data) == list, "Input to property extractor component should be a list of PIL images of objects"

        return self.extract_props(data)

    def train(self, data):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()


class NeuralPropExtractor(_AbsPropExtractor):
    def __init__(self, model):
        super(NeuralPropExtractor, self).__init__()

        self.model = model

    def extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of tuple [(colour, rotation, class)]
        """

        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)

        device = get_device()
        obj_imgs_batch = obj_imgs_batch.to(device)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        preds = [torch.max(pred, dim=1)[1].cpu().numpy() for pred in model_out]
        colours = [COLOURS[idx] for idx in preds[0]]
        rotations = [ROTATIONS[idx] for idx in preds[1]]
        classes = [CLASSES[idx] for idx in preds[2]]

        assert len(colours) == len(rotations) == len(classes), "Number of predictions must be the same"

        props = zip(colours, rotations, classes)
        return props

    def train(self, data):
        pass

    def load(self, path):
        model = load_model(PropertyExtractionModel, path)
        model.eval()
        self.model = model

    def save(self, path):
        save_model(self.model, path)
