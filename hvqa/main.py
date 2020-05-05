import argparse

from hvqa.util.dataset import VideoDataset
from hvqa.util.environment import EnvSpec
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.individually_trained import IndTrainedModel


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

MODEL_PATH = "saved-models/ind-trained"

spec = EnvSpec.from_dict({
    "num_frames": 32,
    "obj_types": [("octopus", False), ("fish", True), ("rock", True), ("bag", True)],
    "properties": {
        "colour": ["red", "blue", "purple", "brown", "green", "silver", "white"],
        "rotation": ["upward-facing", "left-facing", "downward-facing", "right-facing"]
    },
    "relations": ["close"],
    "actions": ["move", "rotate left", "rotate right"],
    "events": ["change colour", "eat a bag", "eat a fish"],
})


def main(train_dir, eval_dir):
    # Create data
    detector = NeuralDetector.load(spec, DETECTOR_PATH)
    train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
    eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=True)

    # Create model
    model = IndTrainedModel.new(spec, err_corr=False, al_model=True)
    model.train(train_data, eval_data)
    model.save(MODEL_PATH)

    # Load model and evaluate again
    model = HardcodedVQAModel.load(MODEL_PATH, spec=spec)
    model.eval_components(eval_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for running VideoQA pipeline on a video")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("eval_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.eval_dir)
