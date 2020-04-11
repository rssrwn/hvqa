import argparse

from hvqa.video_dataset import VideoDataset
from hvqa.models.hardcoded import HardcodedModel


DETECTOR_PATH = "saved-models/detector-e2e-v1_0/after_20_epochs.pt"
PROP_EXTRACTOR_PATH = "saved-models/prop-extractor-v1_0/after_2_epochs.pt"
EVENT_ASP_DIR = "hvqa/events"
QA_ASP_DIR = "hvqa/qa"


def evaluate(model, data):
    total = 0
    num_correct = 0

    for video_idx in range(len(data)):
        print(f"Running on video {video_idx}")
        frames, video_dict = data[video_idx]
        questions = video_dict["questions"]
        q_types = video_dict["question_types"]

        answers = model.run(frames, questions, q_types)
        expected = video_dict["answers"]

        for idx, predicted in enumerate(answers):
            actual = expected[idx]
            if actual == predicted:
                print("correct")
                num_correct += 1
            else:
                print("incorrect")

            total += 1

    acc = (num_correct / total) * 100

    print(f"Num correct: {num_correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {acc:.2}%")


def main(data_dir, model_type):
    data = VideoDataset(data_dir)

    if model_type == "hardcoded":
        model = HardcodedModel(DETECTOR_PATH, PROP_EXTRACTOR_PATH, EVENT_ASP_DIR, QA_ASP_DIR)
    else:
        print("That type of model is not supported")
        return

    evaluate(model, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating full QA pipeline")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_type", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.model_type)
