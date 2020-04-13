import argparse

from hvqa.video_dataset import VideoDataset
from hvqa.models.hardcoded import HardcodedModel


DETECTOR_PATH = "saved-models/detector-e2e-v1_0/after_20_epochs.pt"
PROP_EXTRACTOR_PATH = "saved-models/prop-extractor-v1_0/after_2_epochs.pt"
EVENT_ASP_DIR = "hvqa/events"
QA_ASP_DIR = "hvqa/qa"


def _inc_in_map(coll, key):
    val = coll.get(key)
    if val is None:
        coll[key] = 0

    coll[key] += 1


def evaluate(model, data, verbose=False):
    correct = {}
    incorrect = {}

    print()
    for video_idx in range(len(data)):
        frames, video_dict = data[video_idx]
        questions = video_dict["questions"]
        q_types = video_dict["question_types"]

        answers = model.run(frames, questions, q_types)
        expected = video_dict["answers"]

        video_correct = 0
        for idx, predicted in enumerate(answers):
            actual = expected[idx]
            q_type = q_types[idx]
            if actual == predicted:
                _inc_in_map(correct, q_type)
                video_correct += 1
                if verbose:
                    print(f"Q{idx}: Correct.")

            else:
                _inc_in_map(incorrect, q_type)
                question = questions[idx]
                if verbose:
                    print(f"Q{idx}: Incorrect. Question: {question} -- "
                          f"Answer: Predicted '{predicted}', actual '{actual}'")

        acc = video_correct / len(questions)
        print(f"Video [{video_idx:4}/{len(data):4}] -- {video_correct:2} / {len(questions):2} -- Accuracy: {acc:.0%}")

    q_types = list(set(correct.keys()).union(set(incorrect.keys())))
    sorted(q_types)

    print("Results:")
    print(f"\n{'Question Type':<20}{'Correct':<15}{'Incorrect':<15}Accuracy")
    for q_type in q_types:
        num_correct = correct.get(q_type)
        num_correct = 0 if num_correct is None else num_correct
        num_incorrect = incorrect.get(q_type)
        num_incorrect = 0 if num_incorrect is None else num_incorrect
        acc = (num_correct / (num_correct + num_incorrect))
        print(f"{q_type:<20}{num_correct:<15}{num_incorrect:<15}{acc:.1%}")

    num_correct = sum(correct.values())
    total = num_correct + sum(incorrect.values())
    acc = (num_correct / total) * 100

    print(f"\nNum correct: {num_correct}")
    print(f"Total: {total}")
    print(f"Accuracy: {acc:.3}%\n")

    model.print_timings()


def main(data_dir, model_type):
    data = VideoDataset(data_dir)

    if model_type == "hardcoded":
        model = HardcodedModel(DETECTOR_PATH, PROP_EXTRACTOR_PATH, EVENT_ASP_DIR, QA_ASP_DIR)
    else:
        print("That type of model is not supported")
        return

    evaluate(model, data, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating full QA pipeline")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("model_type", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.model_type)
