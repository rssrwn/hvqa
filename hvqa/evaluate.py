import argparse

from hvqa.spec.env import EnvSpec
from hvqa.util.dataset import VideoDataset
from hvqa.util.dataset import BaselineDataset
from hvqa.detection.detector import NeuralDetector
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.trained import IndTrainedModel
from hvqa.models.baselines.language import RandomAnsModel, BestChoiceModel
from hvqa.models.baselines.neural import (
    LangLstmModel,
    CnnMlpModel,
    CnnObjModel,
    TvqaModel
)


DETECTOR_PATH = "saved-models/detection/v1_0/after_20_epochs.pt"

HARDCODED_MODEL_PATH = "saved-models/hardcoded"
IND_MODEL_PATH = "saved-models/ind-trained"
BEST_CHOICE_MODEL_PATH = "saved-models/best-choice"
LANG_LSTM_MODEL_PATH = "saved-models/lang-lstm"
CNN_MLP_MODEL_PATH = "saved-models/cnn-mlp"
CNN_LSTM_MODEL_PATH = "saved-models/cnn-lstm"
CNN_OBJ_PATH = "saved-models/cnn-obj"
CNN_OBJ_PQ_PATH = "saved-models/cnn-obj-pq"
TVQA_MODEL_PATH = "saved-models/tvqa-sm"
TVQA_CURR_MODEL_PATH = "saved-models/tvqa-curr"

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

ERR_CORR = True
AL_EVENT_MODEL = True

HARDCODED = False


def evaluate(model, data, components=False, verbose=True):
    if components:
        model.eval_components(data)
    else:
        model.eval(data, verbose)


def main(data_dir, model_type, components):
    if model_type == "hardcoded":
        model_path = HARDCODED_MODEL_PATH
        model = HardcodedVQAModel.load(spec, model_path, err_corr=ERR_CORR, al_model=AL_EVENT_MODEL)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED)

    elif model_type == "ind-trained":
        model_path = IND_MODEL_PATH
        model = IndTrainedModel.load(spec, model_path)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED)

    elif model_type == "random":
        model = RandomAnsModel(spec)
        data = BaselineDataset.from_data_dir(data_dir)

    elif model_type == "best-choice":
        model_path = BEST_CHOICE_MODEL_PATH
        model = BestChoiceModel.load(spec, model_path)
        data = BaselineDataset.from_data_dir(data_dir)

    elif model_type == "lang-lstm":
        model_path = LANG_LSTM_MODEL_PATH
        model = LangLstmModel.load(spec, model_path)
        data = BaselineDataset.from_data_dir(data_dir)

    elif model_type == "cnn-mlp":
        model_path = CNN_MLP_MODEL_PATH
        model = CnnMlpModel.load(spec, model_path)
        data = BaselineDataset.from_data_dir(data_dir)

    elif model_type == "cnn-lstm":
        model_path = CNN_LSTM_MODEL_PATH
        model = CnnMlpModel.load(spec, model_path, video_lstm=True)
        data = BaselineDataset.from_data_dir(data_dir)

    elif model_type == "cnn-obj":
        model_path = CNN_OBJ_PATH
        model = CnnObjModel.load(spec, model_path)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED)

    elif model_type == "cnn-obj-pq":
        model_path = CNN_OBJ_PQ_PATH
        model = CnnObjModel.load(spec, model_path, parse_q=True)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED)

    elif model_type == "tvqa":
        model_path = TVQA_MODEL_PATH
        model = TvqaModel.load(spec, model_path)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED, store_frames=True)

    elif model_type == "tvqa-curr":
        model_path = TVQA_CURR_MODEL_PATH
        model = TvqaModel.load(spec, model_path)
        detector = NeuralDetector.load(spec, DETECTOR_PATH)
        data = VideoDataset.from_data_dir(spec, data_dir, detector, hardcoded=HARDCODED, store_frames=True)

    else:
        print("That type of model is not supported")
        return

    evaluate(model, data, components, verbose=True)

    # video_idx = 7
    # frames, video_dict = data[video_idx]
    # visual = Visualiser(model)
    # visual.visualise(frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating full QA pipeline")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_type", type=str)
    parser.add_argument("-c", "--components", action="store_true", help="evaluate components only")
    args = parser.parse_args()
    main(args.data_dir, args.model_type, args.components)
