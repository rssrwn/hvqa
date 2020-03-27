import torch
import torchvision.transforms as T

from hvqa.coordination.knowledge import Obj, Frame, Video


_transform = T.Compose([
    T.ToTensor(),
])


class _AbsDetector:
    def __init__(self):
        pass

    def detect_objs(self, video):
        """
        Detect all objects in a video

        :param video: List of PIL frames
        :return: Video object containing info on objects in video
        """

        raise NotImplementedError


class NeuralDetector(_AbsDetector):
    def __int__(self, model):
        super(NeuralDetector, self).__init__()

        self.model = model

    def detect_objs(self, video):
        imgs_trans = [_transform(img) for img in video]
        imgs_batch = torch.stack(imgs_trans)

        with torch.no_grad():
            detector_out = self.model(imgs_batch)

        frames = []
        for idx, frame in enumerate(detector_out):
            img = video[idx]
            boxes = frame["boxes"]
            labels = frame["labels"]

            objs = []
            for obj_idx, box in enumerate(boxes):
                label = labels[obj_idx]
                obj = Obj(label, box)
                obj.set_image(img)
                objs.append(obj)

            frame = Frame(objs)
            frames.append(frame)

        video = Video(frames)
        return video
