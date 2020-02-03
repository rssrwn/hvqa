import argparse

from hvqa.detection.evaluation import DetectionEvaluator, ClassificationEvaluator


DEFAULT_CONF_THRESHOLD = 0.5


def eval_classifier(evaluator, model_file, threshold):
    evaluator.eval_models(model_file, threshold)


def eval_detector(evaluator, model_file, threshold, visualise):
    if visualise:
        evaluator.visualise(model_file, threshold)

    # TODO uncomment
    # evaluator.eval_model(model_file, threshold)


def main(test_dir, model_file, threshold, classifier, visualise):
    if not threshold:
        threshold = DEFAULT_CONF_THRESHOLD

    if classifier:
        print("Evaluating classifier performance...")
        evaluator = ClassificationEvaluator(test_dir)
        eval_classifier(evaluator, model_file, threshold)
    else:
        print("Evaluating detector performance...")
        evaluator = DetectionEvaluator(test_dir, "saved-models/resnet-classifier-v1/after_10_epochs.pt")
        eval_detector(evaluator, model_file, threshold, visualise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("-c", "--classifier", action="store_true", default=False)
    parser.add_argument("-v", "--visualise", action="store_true", default=False)
    args = parser.parse_args()
    main(args.eval_dir, args.model_file, args.threshold, args.classifier, args.visualise)
