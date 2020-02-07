import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ImageDraw
import numpy as np

from hvqa.util import UnknownObjectTypeException, get_device

from lib.vision.engine import evaluate


NUM_IMAGES = 3


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class DetectionEvaluator(_AbsEvaluator):
    def __init__(self, test_loader):
        super(DetectionEvaluator, self).__init__(test_loader)

    def eval_model(self, model, threshold=None):
        device = get_device()
        model.eval()
        evaluate(model, self.test_loader, device)

    def visualise(self, model, conf_threshold=0.5):
        dataset = self.test_loader.dataset
        transform = self.test_loader.dataset.transforms

        model.eval()

        num_images = len(dataset)
        for i in range(NUM_IMAGES):
            image_idx = np.random.randint(0, num_images)

            img_tensor, frame_dict = dataset.get_image(image_idx)

            if transform:
                img_tensor = transform(img_tensor)

            img = T.ToPILImage()(img_tensor)

            # Add actual bboxs
            draw = ImageDraw.Draw(img)
            self._add_bboxs(draw, [obj["position"] for obj in frame_dict["objects"]])

            # Get network output
            with torch.no_grad():
                net_out = model(img_tensor[None, :, :, :])

            scores = net_out[0]["scores"]
            idxs = scores > conf_threshold
            boxes = list(net_out[0]["boxes"][idxs, :].numpy())
            labels = list(net_out[0]["labels"][idxs].numpy())
            short_labels = [self._shorten_label(label) for label in labels]

            # Add predicted bboxs and labels
            self._add_bboxs(draw, boxes, ground_truth=False)
            for idx, label in enumerate(short_labels):
                x1, y1, x2, y2 = boxes[idx]
                self._add_labels(draw, (x1, y1), label)

            img.show()

    @staticmethod
    def _add_bboxs(drawer, positions, ground_truth=True):
        colour = "blue" if ground_truth else "red"
        for position in positions:
            x1, y1, x2, y2 = position
            x1 = round(x1) - 1
            y1 = round(y1) - 1
            x2 = round(x2) + 1
            y2 = round(y2) + 1
            drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour)

    @staticmethod
    def _add_labels(drawer, position, text):
        colour = "red"
        x1, y1 = position
        y1 -= 10
        drawer.text((x1, y1), text, fill=colour)

    @staticmethod
    def _shorten_label(label):
        if label == 0:
            return "back"
        elif label == 1:
            return "o"
        elif label == 2:
            return "f"
        elif label == 3:
            return "b"
        elif label == 4:
            return "r"
        else:
            raise UnknownObjectTypeException(f"Unknown label {label}")


class ClassificationEvaluator(_AbsEvaluator):
    def __init__(self, test_loader):
        super(ClassificationEvaluator, self).__init__(test_loader)

    def eval_model(self, model, threshold=0.7):
        print("Evaluating classification model")

        device = get_device()
        model.eval()

        mses = torch.tensor([], dtype=torch.float32)
        num_correct = torch.tensor([], dtype=torch.float32)
        for i, (x, y) in enumerate(self.test_loader):
            x = x.to(device=device)

            with torch.no_grad():
                preds = model(x)

            num_predictions = preds.shape[1]

            preds = preds.to("cpu")

            # Calculate MSE
            mse = F.mse_loss(preds, y, reduction="none")
            mse_sum = torch.sum(mse, 1)
            mses = torch.cat((mses, mse_sum))

            # Calculate number correct predictions
            act_bool = y == 1
            preds_bool = preds > threshold
            correct = act_bool == preds_bool
            num_correct_batch = torch.sum(correct, 1).type(torch.float32)
            num_correct = torch.cat((num_correct, num_correct_batch))

        assert mses.shape[0] == num_correct.shape[0], "Different numbers of images seen"

        print(f"Avg MSE: {torch.mean(mses):.4f}")
        print(f"Avg correct predictions: {torch.mean(num_correct):.4f} at threshold {threshold}")
        print(f"Avg prediction accuracy: {torch.mean(num_correct) / num_predictions:.4f} at threshold {threshold}")
