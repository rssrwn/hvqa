import torch
import torchvision.transforms as T

from hvqa.util.definitions import COLOURS, ROTATIONS, CLASSES


_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])


class _AbsPropExtractor:
    def __init__(self):
        pass

    def extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of tuple [(colour, rotation, class)]
        """

        raise NotImplementedError


class NeuralPropExtractor(_AbsPropExtractor):
    def __init__(self, model):
        super(NeuralPropExtractor, self).__init__()

        self.model = model

    def extract_props(self, obj_imgs):
        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        preds = [torch.max(pred, dim=1)[1].numpy() for pred in model_out]
        colours = [COLOURS[idx] for idx in preds[0]]
        rotations = [ROTATIONS[idx] for idx in preds[1]]
        classes = [CLASSES[idx] for idx in preds[2]]

        assert len(colours) == len(rotations) == len(classes), "Number of predictions must be the same"

        props = zip(colours, rotations, classes)
        return props
