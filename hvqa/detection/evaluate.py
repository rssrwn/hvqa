import argparse
import torch
from torch.utils.data import DataLoader

from hvqa.util.definitions import detector_transforms
from hvqa.util.util import collate_func, get_device
from hvqa.detection.evaluation import DetectionEvaluator, ClassificationEvaluator
from hvqa.detection.models import ClassifierModel, DetectionBackboneWrapper, DetectionModel, DetectionBackbone
from hvqa.detection.dataset import DetectionDataset


DEFAULT_CONF_THRESHOLD = 0.5


def eval_classifier(evaluator, model_file, threshold):
    evaluator.eval_models(model_file, threshold)


def eval_detector(evaluator, model, threshold, visualise):
    if visualise:
        evaluator.visualise(model, threshold)

    # TODO uncomment
    # evaluator.eval_model(model, threshold)


def main(test_dir, model_file, threshold, classifier, visualise):
    if not threshold:
        threshold = DEFAULT_CONF_THRESHOLD

    if classifier:
        print("Evaluating classifier performance...")
        evaluator = ClassificationEvaluator(test_dir)
        eval_classifier(evaluator, model_file, threshold)
    else:
        print("Evaluating detector performance...")

        device = get_device()

        # pretrained = ClassifierModel()
        # pretrained.load_state_dict(torch.load("saved-models/resnet-classifier-v1/after_10_epochs.pt", map_location=device))
        # backbone = DetectionBackboneWrapper(pretrained)

        backbone = DetectionBackbone()
        model = DetectionModel(backbone)
        model.load_state_dict(torch.load(model_file, map_location=device))

        dataset_test = DetectionDataset(test_dir, transforms=detector_transforms)
        loader_test = DataLoader(dataset_test, batch_size=48, shuffle=True, collate_fn=collate_func)

        evaluator = DetectionEvaluator(loader_test)
        eval_detector(evaluator, model, threshold, visualise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("-c", "--classifier", action="store_true", default=False)
    parser.add_argument("-v", "--visualise", action="store_true", default=False)
    args = parser.parse_args()
    main(args.eval_dir, args.model_file, args.threshold, args.classifier, args.visualise)
