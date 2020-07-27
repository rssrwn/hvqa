import argparse

from hvqa.util.dataset import VideoDataset, BaselineDataset
from hvqa.spec.env import EnvSpec
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.individually_trained import IndTrainedModel
from hvqa.models.baselines.language import BestChoiceModel, LstmModel


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

HARDCODED_MODEL_PATH = "saved-models/hardcoded"
IND_MODEL_PATH = "saved-models/ind-trained"
BEST_CHOICE_MODEL_PATH = "saved-models/best-choice"
LANG_LSTM_MODEL_PATH = "saved-models/lang-lstm"

spec = EnvSpec.from_dict({
    "num_frames": 32,
    "obj_types": [("octopus", False), ("fish", True), ("rock", True), ("bag", True)],
    "properties": {
        "colour": ["red", "blue", "purple", "brown", "green", "silver", "white"],
        "rotation": ["upward-facing", "right-facing", "downward-facing", "left-facing"]
    },
    "relations": ["close"],
    "actions": ["move", "rotate left", "rotate right", "nothing"],
    "effects": ["change colour", "eat a bag", "eat a fish"],
})


def main(train_dir, eval_dir, model_type):
    detector = NeuralDetector.load(spec, DETECTOR_PATH)

    if model_type == "hardcoded":
        model_path = HARDCODED_MODEL_PATH
        model = HardcodedVQAModel.new(spec)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=True)

    elif model_type == "ind-trained":
        model_path = IND_MODEL_PATH
        model = IndTrainedModel.new(spec)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=True)

    elif model_type == "best-choice":
        model_path = BEST_CHOICE_MODEL_PATH
        model = BestChoiceModel(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "lang-lstm":
        model_path = LANG_LSTM_MODEL_PATH
        model = LstmModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    else:
        print("That model type is not supported")
        return

    model.train(train_data, eval_data)
    model.save(model_path)

    # Load model and evaluate again
    # model = IndTrainedModel.load(spec, IND_MODEL_PATH)
    # model.eval(eval_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training H-PERL model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_type", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.eval_dir, args.model_type)
