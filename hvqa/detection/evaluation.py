import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from shapely.geometry import Polygon
from PIL import ImageDraw
import numpy as np

from hvqa.util import load_model, extract_bbox_and_class, get_device
from hvqa.detection.dataset import DetectionDataset, ClassificationDataset
from hvqa.detection.models import DetectionModel, ClassifierModel, DetectionBackbone

from lib.vision.engine import evaluate


class _AbsEvaluator:
    def __init__(self, test_loader):
        # TODO transforms
        self.test_loader = test_loader

    # def eval_models_from_file(self, model_dir, conf_threshold=0.5):
    #     model_path = Path(model_dir)
    #     if model_path.is_dir():
    #         print("Model path is a directory, evaluating all models in dir...")
    #         for model_file in model_path.iterdir():
    #             self.eval_model_from_file(model_file, conf_threshold)
    #     else:
    #         print("Model path is a file, evaluating single model...")
    #         self.eval_model_from_file(model_dir, conf_threshold)

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class DetectionEvaluator(_AbsEvaluator):
    def __init__(self, test_loader):
        super(DetectionEvaluator, self).__init__(test_loader)

    def eval_model(self, model, threshold=None):
        device = get_device()
        evaluate(model, self.test_loader, device)

    @staticmethod
    def _create_shape(bbox):
        x1, y1, x2, y2 = bbox
        return Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    @staticmethod
    def _calc_iou(bbox1, bbox2):
        poly_1 = DetectionEvaluator._create_shape(bbox1)
        poly_2 = DetectionEvaluator._create_shape(bbox2)
        union = poly_1.union(poly_2).area
        if union == 0:
            return 0

        return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

    def visualise(self, model, conf_threshold=0.5):
        dataset = self.test_loader.dataset

        # Collect image
        num_images = len(dataset)
        image_idx = np.random.randint(0, num_images)
        img, frame_dict = dataset.get_image(image_idx)

        # Add bboxs
        draw = ImageDraw.Draw(img)
        self._add_bboxs(draw, [obj["position"] for obj in frame_dict["objects"]])

        model.eval()

        # TODO use transforms
        img_arr = np.transpose(np.asarray(img, dtype=np.float32) / 255, (2, 0, 1))
        img_tensor = torch.from_numpy(img_arr)

        with torch.no_grad():
            net_out = model(img_tensor[None, :, :, :])

        # TODO make work with batch (atm assume single elem in list)
        scores = net_out[0]["scores"]
        idxs = scores > conf_threshold

        boxes = list(net_out[0]["boxes"][idxs, :].numpy())
        labels = list(net_out[0]["labels"][idxs].numpy())

        # preds = extract_bbox_and_class(net_out[0, :, :, :], conf_threshold)
        DetectionEvaluator._add_bboxs(draw, boxes, ground_truth=False)

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


class ClassificationEvaluator(_AbsEvaluator):
    def __init__(self, test_data_dir):
        super(ClassificationEvaluator, self).__init__(test_data_dir)
        batch_size = 48
        dataset = ClassificationDataset(test_data_dir)
        self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def eval_model(self, model, threshold=0.7):
        print("Evaluating classification model")

        model.eval()

        mses = torch.tensor([], dtype=torch.float32)
        num_correct = torch.tensor([], dtype=torch.float32)
        for i, (x, y) in enumerate(self.test_loader):
            with torch.no_grad():
                preds = model(x)

            num_predictions = preds.shape[1]

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

        print(f"Avg MSE: {torch.mean(mses)}")
        print(f"Avg correct predictions: {torch.mean(num_correct)} at threshold {threshold}")
        print(f"Avg prediction accuracy: {torch.mean(num_correct) / num_predictions} at threshold {threshold}")
