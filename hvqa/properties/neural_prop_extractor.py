import torch
import torchvision.transforms as T

from hvqa.properties.models import PropertyExtractionModel
from hvqa.util.func import get_device, load_model, save_model
from hvqa.util.interfaces import Component


_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


class NeuralPropExtractor(Component):
    def __init__(self, spec, model):
        super(NeuralPropExtractor, self).__init__()

        self.spec = spec
        self.model = model

    def run_(self, video):
        for frame in video.frames:
            objs = frame.objs
            obj_imgs = [obj.img for obj in objs]
            props = self._extract_props(obj_imgs)
            for idx, prop_vals in enumerate(props):
                obj = objs[idx]
                for prop_idx, val in enumerate(prop_vals):
                    prop_name = self.spec.prop_names[prop_idx]
                    obj.set_prop_val(prop_name, val)

    def _extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of tuple [(prop_val1, prop_val2, ...)]
        """

        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)

        device = get_device()
        obj_imgs_batch = obj_imgs_batch.to(device)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        preds = [torch.max(pred, dim=1)[1].cpu().numpy() for pred in model_out]

        prop_list = []
        length = None
        for prop_idx, pred in enumerate(preds):
            vals = [self.spec.prop_values[idx] for idx in pred]
            length = len(vals) if length is None else length
            assert length == len(vals), "Number of predictions must be the same"
            prop_list.append(vals)

        props = zip(prop_list)
        return props

    def train(self, data):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        model = PropertyExtractionModel(spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    @staticmethod
    def load(path, spec):
        model = load_model(PropertyExtractionModel, path, (spec,))
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    def save(self, path):
        save_model(self.model, path)
