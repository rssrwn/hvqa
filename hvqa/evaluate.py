import argparse

from hvqa.spec.env import EnvSpec
from hvqa.util.dataset import VideoDataset
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.individually_trained import IndTrainedModel


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

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

ERR_CORR = True
AL_EVENT_MODEL = True

ERROR_PROB = 0


def evaluate(model, data, verbose=True):
    model.eval(data, verbose)


def main(data_dir, model_type):
    detector = NeuralDetector.load(spec, DETECTOR_PATH)
    data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=False, err_prob=ERROR_PROB)

    if model_type == "hardcoded":
        model_path = "saved-models/hardcoded"
        model = HardcodedVQAModel.load(spec, model_path, err_corr=ERR_CORR, al_model=AL_EVENT_MODEL)
    elif model_type == "ind-trained":
        model_path = "saved-models/ind-trained"
        model = IndTrainedModel.load(spec, model_path)
    else:
        print("That type of model is not supported")
        return

    evaluate(model, data, verbose=True)

    # video_idx = 7
    # frames, video_dict = data[video_idx]
    # visual = Visualiser(model)
    # visual.visualise(frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating full QA pipeline")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_type", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.model_type)
