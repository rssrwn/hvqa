import argparse

from hvqa.detection.evaluator import DetectionEvaluator


CONF_THRESHOLD = 0.5


def main(test_dir, model_file, visualise):
    evaluator = DetectionEvaluator(test_dir)

    if visualise:
        evaluator.visualise(model_file, CONF_THRESHOLD)

    # evaluator.eval_model(model_file, CONF_THRESHOLD)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate performance of a trained object detection model")
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_file", type=str)
    parser.add_argument("-v", "--visualise", action="store_true", default=False)
    args = parser.parse_args()
    main(args.eval_dir, args.model_file, args.visualise)
