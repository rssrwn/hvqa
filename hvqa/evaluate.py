import argparse

from hvqa.video_dataset import VideoDataset
from hvqa.models.hardcoded import HardcodedVQAModel
from hvqa.models.visualise import Visualiser


DETECTOR_PATH = "saved-models/detector-e2e-v1_0/after_20_epochs.pt"
PROP_EXTRACTOR_PATH = "saved-models/prop-extractor-v1_0/after_2_epochs.pt"
EVENT_ASP_DIR = "hvqa/events"
QA_ASP_DIR = "hvqa/qa"
ERR_CORR = False


def evaluate(model, data, verbose=True):
    model.eval(data, verbose)


def main(data_dir, model_type):
    data = VideoDataset(data_dir)

    if model_type == "hardcoded":
        model = HardcodedVQAModel(EVENT_ASP_DIR, QA_ASP_DIR, ERR_CORR, DETECTOR_PATH, PROP_EXTRACTOR_PATH)
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
