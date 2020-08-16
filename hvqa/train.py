import argparse

from hvqa.util.dataset import VideoDataset, BaselineDataset
from hvqa.spec.env import EnvSpec
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.trained import IndTrainedModel
from hvqa.models.baselines.language import BestChoiceModel
from hvqa.models.baselines.neural import (
    LangLstmModel,
    CnnMlpModel,
    PropRelModel,
    EventModel,
    CnnMlpPreModel,
    CnnObjModel,
    PropRelObjModel,
    EventObjModel
)


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

HARDCODED_MODEL_PATH = "saved-models/hardcoded"
IND_MODEL_PATH = "saved-models/ind-trained"
BEST_CHOICE_MODEL_PATH = "saved-models/best-choice"
LANG_LSTM_MODEL_PATH = "saved-models/lang-lstm"
CNN_MLP_MODEL_PATH = "saved-models/cnn-mlp"
CNN_LSTM_MODEL_PATH = "saved-models/cnn-lstm"
PROP_REL_MODEL_PATH = "saved-models/pre/prop-rel"
EVENT_MODEL_PATH = "saved-models/pre/event"
CNN_MLP_PRE_PATH = "saved-models/cnn-mlp-pre"
CNN_MLP_PRE_PQ_PATH = "saved-models/cnn-mlp-pre-pq"
CNN_OBJ_PATH = "saved-models/cnn-obj"
CNN_OBJ_ATT_PATH = "saved-models/cnn-obj-att"
PROP_REL_OBJ_PATH = "saved-models/pre/prop-rel-obj"
EVENT_OBJ_PATH = "saved-models/pre/event-obj"

spec = EnvSpec.from_dict({
    "num_frames": 32,
    "obj_types": [("octopus", False), ("fish", True), ("rock", True), ("bag", True)],
    "properties": {
        "colour": ["red", "blue", "purple", "brown", "green", "silver", "white"],
        "rotation": ["upward-facing", "right-facing", "downward-facing", "left-facing"]
    },
    "relations": ["close", "above", "below"],
    "actions": ["move", "rotate left", "rotate right", "nothing"],
    "effects": ["change colour", "eat a bag", "eat a fish"],
})


def main(train_dir, eval_dir, model_type):
    if model_type == "hardcoded":
        model_path = HARDCODED_MODEL_PATH
        model = HardcodedVQAModel.new(spec)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=True)

    elif model_type == "ind-trained":
        model_path = IND_MODEL_PATH
        model = IndTrainedModel.new(spec)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=True)

    elif model_type == "best-choice":
        model_path = BEST_CHOICE_MODEL_PATH
        model = BestChoiceModel(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "lang-lstm":
        model_path = LANG_LSTM_MODEL_PATH
        model = LangLstmModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "cnn-mlp":
        model_path = CNN_MLP_MODEL_PATH
        model = CnnMlpModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "cnn-lstm":
        model_path = CNN_LSTM_MODEL_PATH
        model = CnnMlpModel.new(spec, video_lstm=True)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "prop-rel":
        model_path = PROP_REL_MODEL_PATH
        model = PropRelModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "event":
        model_path = EVENT_MODEL_PATH
        model = EventModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "cnn-mlp-pre":
        model_path = CNN_MLP_PRE_PATH
        model = CnnMlpPreModel.new(spec)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "cnn-mlp-pre-pq":
        model_path = CNN_MLP_PRE_PQ_PATH
        model = CnnMlpPreModel.new(spec, parse_q=True)
        train_data = BaselineDataset.from_data_dir(train_dir)
        eval_data = BaselineDataset.from_data_dir(eval_dir)

    elif model_type == "cnn-obj":
        model_path = CNN_OBJ_PATH
        model = CnnObjModel.new(spec)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=False)

    elif model_type == "cnn-obj-att":
        model_path = CNN_OBJ_ATT_PATH
        model = CnnObjModel.new(spec, att=True)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=False)

    elif model_type == "prop-rel-obj":
        model_path = PROP_REL_OBJ_PATH
        model = PropRelObjModel.new(spec)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=False)

    elif model_type == "event-obj":
        model_path = EVENT_OBJ_PATH
        model = EventObjModel.new(spec)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        train_data = VideoDataset.from_data_dir(spec, train_dir, detector, hardcoded=False)
        eval_data = VideoDataset.from_data_dir(spec, eval_dir, detector, hardcoded=False)

    else:
        print("That model type is not supported")
        return

    model.train(train_data, eval_data, save_path=model_path)
    model.save(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for training H-PERL model")
    parser.add_argument("train_dir", type=str)
    parser.add_argument("eval_dir", type=str)
    parser.add_argument("model_type", type=str)
    args = parser.parse_args()
    main(args.train_dir, args.eval_dir, args.model_type)
