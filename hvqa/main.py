import argparse

from hvqa.util.dataset import VideoDataset
from hvqa.util.environment import EnvSpec
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

MODEL_PATH = "saved-models/hardcoded"

spec = EnvSpec.from_dict({
    "num_frames": 32,
    "obj_types": [("octopus", False), ("fish", True), ("rock", True), ("bag", True)],
    "properties": {
        "colour": ["red", "blue", "purple", "brown", "green", "silver", "white"],
        # "rotation": ["upward-facing", "left-facing", "downward-facing", "right-facing"]
        "rotation": [0, 1, 2, 3]
    },
    "relations": ["close to"],
    "actions": ["move", "rotate left", "rotate right"],
    "events": ["change colour", "eat a bag", "eat a fish"],
})


def main(train_dir, eval_dir):
    train_data = VideoDataset(spec, train_dir, hardcoded=True)
    train_data = [train_data[idx] for idx in range(len(train_data))]
    eval_data = VideoDataset(spec, eval_dir, hardcoded=True)
    eval_data = [eval_data[idx] for idx in range(len(eval_data))]
    detector = NeuralDetector.load(DETECTOR_PATH)
    model = HardcodedVQAModel.new(spec, detector)
    model.train(train_data, eval_data)
    model.save(MODEL_PATH)

    model = HardcodedVQAModel.load(MODEL_PATH)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for running VideoQA pipeline on a video")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("eval_dir", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.eval_dir)
