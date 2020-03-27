import torch
import torchvision.transforms as T

from hvqa.coordination.knowledge import Obj, Frame, Video


_transform = T.Compose([
    T.ToTensor(),
])


class _AbsDetector:
    def __init__(self):
        pass

    def detect_objs(self, frames):
        """
        Detect all objects in a video

        :param frames: List of PIL frames
        :return: Video object containing info on objects in video
        """

        raise NotImplementedError


class NeuralDetector(_AbsDetector):
    def __int__(self, model):
        super(NeuralDetector, self).__init__()

        self.model = model

    def detect_objs(self, frames):
        imgs_trans = [_transform(img) for img in frames]
        imgs_batch = torch.stack(imgs_trans)

        with torch.no_grad():
            detector_out = self.model(imgs_batch)

        frames_objs = []
        for idx, frame in enumerate(detector_out):
            img = frames[idx]
            bboxs = [tuple(map(round, bbox)) for bbox in list(frame["boxes"])]
            labels = list(frame["labels"])

            objs = []
            for obj_idx, bbox in enumerate(bboxs):
                label = labels[obj_idx].numpy()
                obj = Obj(label, bbox.numpy())
                obj.set_image(img)
                objs.append(obj)

            frame = Frame(objs)
            frames_objs.append(frame)

        video = Video(frames_objs)
        return video
