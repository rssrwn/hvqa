import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from pathlib import Path
from shapely.geometry import Polygon
from PIL import ImageDraw
import numpy as np

from hvqa.util import build_data_loader, load_model, extract_bbox_and_class
from hvqa.detection.dataset import DetectionDataset, ClassifierDataset
from hvqa.detection.model import DetectionModel, MyResNet


class _AbsEvaluator:
    def __init__(self, test_data_dir):
        self.test_data_dir = test_data_dir

    def eval_models(self, model_dir, conf_threshold=0.5):
        model_path = Path(model_dir)
        if model_path.is_dir():
            print("Model path is a directory, evaluating all models in dir...")
            for model_file in model_path.iterdir():
                self.eval_model(model_file, conf_threshold)
        else:
            print("Model path is a file, evaluating single model...")
            self.eval_model(model_dir, conf_threshold)

    def eval_model(self, model_file, threshold):
        raise NotImplementedError()


class DetectionEvaluator(_AbsEvaluator):
    def __init__(self, test_data_dir):
        super(DetectionEvaluator, self).__init__(test_data_dir)
        batch_size = 1
        self.test_loader = build_data_loader(DetectionDataset, test_data_dir, batch_size)

    def eval_model(self, model_file, threshold=0.5):
        model = load_model(DetectionModel, Path(model_file))
        model.eval()

        ious = []
        for i, (x, y) in enumerate(self.test_loader):
            with torch.no_grad():
                preds = model(x)

            preds_arr = extract_bbox_and_class(preds[0, :, :, :], threshold)
            actual_arr = extract_bbox_and_class(y[0, :, :, :], threshold)

            for actual_obj in actual_arr:
                best_iou = None
                for obj_idx, pred_obj in enumerate(preds_arr):
                    iou = self._calc_iou(actual_obj[0], pred_obj[0])
                    if best_iou is None or abs(1 - iou) < abs(1 - best_iou):
                        best_iou = iou

                ious.append(best_iou)

        avg_iou = sum(ious) / len(ious)
        print(f"Avg IOU: {avg_iou}")

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

    def visualise(self, model_file, conf_threshold=0.5):
        dataset = self.test_loader.dataset

        # Collect image
        num_images = len(dataset)
        image_idx = np.random.randint(0, num_images)
        img, frame_dict = dataset.get_image(image_idx)

        # Add bboxs
        draw = ImageDraw.Draw(img)
        self._add_bboxs(draw, [obj["position"] for obj in frame_dict["objects"]])

        model = load_model(DetectionModel, Path(model_file))
        img_arr = np.transpose(np.asarray(img, dtype=np.float32) / 255, (2, 0, 1))
        img_tensor = torch.from_numpy(img_arr)

        with torch.no_grad():
            net_out = model(img_tensor[None, :, :, :])

        preds = extract_bbox_and_class(net_out[0, :, :, :], conf_threshold)
        DetectionEvaluator._add_bboxs(draw, [pred[0] for pred in preds], ground_truth=False)

        img.show()

    @staticmethod
    def _add_bboxs(drawer, positions, ground_truth=True):
        colour = "blue" if ground_truth else "red"
        for position in positions:
            x1, y1, x2, y2 = position
            x1 -= 1
            y1 -= 1
            x2 += 1
            y2 += 1
            drawer.rectangle((x1, y1, x2, y2), fill=None, outline=colour)


class ClassificationEvaluator(_AbsEvaluator):
    def __init__(self, test_data_dir):
        super(ClassificationEvaluator, self).__init__(test_data_dir)
        batch_size = 48
        img_size = 128
        dataset = ClassifierDataset(test_data_dir, img_size)
        self.test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def eval_model(self, model_file, threshold=0.7):
        model = load_model(MyResNet, Path(model_file))
        model.eval()

        print(f"Evaluating model at {model_file} ...")

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
        print(f"Avg correct predictions: {torch.mean(num_correct)}")
        print(f"Avg prediction accuracy: {torch.mean(num_correct) / num_predictions}")
