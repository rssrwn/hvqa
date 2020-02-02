import torch
from pathlib import Path
from shapely.geometry import Polygon
from PIL import ImageDraw
import numpy as np

from hvqa.util import build_data_loader, load_model, extract_bbox_and_class
from hvqa.detection.dataset import DetectionDataset
from hvqa.detection.model import DetectionModel


class DetectionEvaluator:
    def __init__(self, test_data_dir):
        batch_size = 1
        self.test_loader = build_data_loader(DetectionDataset, test_data_dir, batch_size)

    def eval_models(self, model_dir, conf_threshold=0.5):
        for model_file in Path(model_dir).iterdir():
            self.eval_model(model_file, conf_threshold)

    def eval_model(self, model_file, conf_threshold=0.5):
        model = load_model(DetectionModel, Path(model_file))
        model.eval()

        ious = []
        for i, (x, y) in enumerate(self.test_loader):
            with torch.no_grad():
                preds = model(x)

            preds_arr = extract_bbox_and_class(preds[0, :, :, :], conf_threshold)
            actual_arr = extract_bbox_and_class(y[0, :, :, :], conf_threshold)

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
