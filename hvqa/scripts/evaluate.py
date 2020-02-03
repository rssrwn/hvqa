import argparse

from hvqa.detection.evaluator import DetectionEvaluator, ClassificationEvaluator


CONF_THRESHOLD = 0.5


def eval_classifier(evaluator, model_file):
    evaluator.eval_models(model_file)


def eval_detector(evaluator, model_file, visualise):
    if visualise:
        evaluator.visualise(model_file, CONF_THRESHOLD)

    # evaluator.eval_model(model_file, CONF_THRESHOLD)


def main(test_dir, model_file, classifier, visualise):
    if classifier:
        evaluator = ClassificationEvaluator(test_dir)
        eval_classifier(evaluator, model_file)
    else:
        evaluator = DetectionEvaluator(test_dir)
        eval_detector(evaluator, model_file, visualise)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("-c", "--classifier", action="store_true", default=False)
    parser.add_argument("-v", "--visualise", action="store_true", default=False)
    args = parser.parse_args()
    main(args.eval_dir, args.model_file, args.classifier, args.visualise)
