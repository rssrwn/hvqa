import argparse
from shapely.geometry import Polygon
from pathlib import Path

from hvqa.util import *
from hvqa.detection.dataset import DetectionBatchDataset
from hvqa.detection.model import DetectionModel


CONF_THRESHOLD = 0.5


def _resize_bbox(bbox):
    x1, y1, x2, y2 = bbox
    x1 = round(x1 * IMG_SIZE)
    y1 = round(y1 * IMG_SIZE)
    x2 = round(x2 * IMG_SIZE)
    y2 = round(y2 * IMG_SIZE)
    return x1, y1, x2, y2


def _extract_bbox_and_class(img_out):
    preds = img_out[:, img_out[4, :, :] > CONF_THRESHOLD]

    preds_arr = []
    for pred_idx in range(preds.shape[1]):
        pred = preds[:, pred_idx]
        bbox = _resize_bbox(pred[0:4].numpy())
        conf, idx = torch.max(pred[5:], 0)
        preds_arr.append([bbox, conf.item(), idx.item()])

    return preds_arr


def _create_shape(bbox):
    x1, y1, x2, y2 = bbox
    return Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _calc_iou(bbox1, bbox2):
    poly_1 = _create_shape(bbox1)
    poly_2 = _create_shape(bbox2)
    union = poly_1.union(poly_2).area
    if union == 0:
        return 0

    return poly_1.intersection(poly_2).area / poly_1.union(poly_2).area


def _eval_model(loader, model):
    ious = []
    for i, (x, y) in enumerate(loader):
        print(f"Frame number: {i}")

        with torch.no_grad():
            preds = model(x)

        preds_arr = _extract_bbox_and_class(preds[0, :, :, :])
        actual_arr = _extract_bbox_and_class(y[0, :, :, :])

        for actual_obj in actual_arr:
            best_iou = None
            for obj_idx, pred_obj in enumerate(preds_arr):
                iou = _calc_iou(actual_obj[0], pred_obj[0])
                if best_iou is None or abs(1 - iou) < abs(1 - best_iou):
                    best_iou = iou

            ious.append(best_iou)

    return ious


def evaluate_models(loader, models, names):
    for idx, model in enumerate(models):
        name = names[idx]
        model.eval()
        ious = _eval_model(loader, model)
        avg_iou = sum(ious) / len(ious)
        print(f"Model {name}: avg iou: {avg_iou}")


def load_models(model_path):
    models = []
    files = []

    # Allow directories
    if model_path.is_dir():
        for model_file in model_path.iterdir():
            models.append(load_obj_detection_model(model_file))
            files.append(model_file)

    elif model_path.exists():
        models.append(load_obj_detection_model(model_path))
        files.append(model_path)

    else:
        raise FileNotFoundError(f"Either {model_path} does not exist or does not contain any model files")

    return models, files


def load_obj_detection_model(path):
    """
    Load a model whose state_dict has been saved
    Note: This does work for models whose entire object has been saved

    :param path: Path to model params
    :return: Model object
    """

    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = DetectionModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main(test_dir, model_file):
    test_path = Path(test_dir)
    model_path = Path(model_file)
    models, names = load_models(model_path)
    test_loader = build_data_loader(DetectionBatchDataset, test_path, 1)
    evaluate_models(test_loader, models, names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    main(args.eval_dir, args.model_path)
