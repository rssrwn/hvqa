# Abstract QA Model Class File

import time

from hvqa.util.interfaces import Model
from hvqa.util.func import inc_in_map


class _AbsVQAModel(Model):
    """
    Abstract QA Model Class
    Most of the models will run the same pipeline which is contained in this class
    Models need to set the components within their __init__ functions
    Models can also override methods if required
    """

    def __init__(self, properties, tracker, relations, events, qa):
        self.prop_classifier = properties
        self.tracker = tracker
        self.relation_classifier = relations
        self.event_detector = events
        self.qa_system = qa

        self.components = [
            self.prop_classifier,
            self.tracker,
            self.relation_classifier,
            self.event_detector,
            self.qa_system
        ]

        self._eval_timings = {
            "Detector": 0,
            "Properties": 0,
            "Tracker": 0,
            "Relations": 0,
            "Events": 0,
            "QA": 0
        }

    def run(self, video):
        """
        Generate answers to given questions

        :param video: Video obj
        :return: Answers: [str]
        """

        self.process(video)
        self._time_func(self.qa_system.run_, (video,), "QA")
        answers = video.answers
        return answers

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError("AbsModel is abstract; objects should not be created")

    @staticmethod
    def new(spec, detector, **kwargs):
        raise NotImplementedError("AbsModel is abstract; objects should not be created")

    @staticmethod
    def load(path, **kwargs):
        raise NotImplementedError("AbsModel is abstract; objects should not be created")

    def save(self, path):
        raise NotImplementedError("AbsModel is abstract; objects should not be created")

    def eval(self, data, verbose=True):
        """
        Evaluate the performance of the model
        Will print the results incrementally and as a final tally

        :param data: VideoDataset obj for evaluation data
        :param verbose: Boolean
        """

        assert not data.is_hardcoded(), "Dataset must not be hardcoded when evaluating"

        self._eval_timings["Detector"] = data.detector_timing()
        start_time = time.time()

        correct, incorrect = self._eval_videos(data, verbose)
        q_types = list(set(correct.keys()).union(set(incorrect.keys())))
        sorted(q_types)

        print("\nResults:")
        print(f"{'Question Type':<20}{'Correct':<15}{'Incorrect':<15}Accuracy")
        for q_type in q_types:
            num_correct = correct.get(q_type)
            num_correct = 0 if num_correct is None else num_correct
            num_incorrect = incorrect.get(q_type)
            num_incorrect = 0 if num_incorrect is None else num_incorrect
            acc = (num_correct / (num_correct + num_incorrect))
            print(f"{q_type:<20}{num_correct:<15}{num_incorrect:<15}{acc:.1%}")

        num_correct = sum(correct.values())
        total = num_correct + sum(incorrect.values())
        acc = (num_correct / total)

        print(f"\nNum correct: {num_correct}")
        print(f"Total: {total}")
        print(f"Accuracy: {acc:.1%}\n")

        self.print_timings()

        end_time = time.time()
        print(f"Total time: {end_time - start_time:.1f} seconds.\n")

    def _eval_videos(self, data, verbose):
        correct = {}
        incorrect = {}

        print()
        for video_idx in range(len(data)):
            video, expected = data[video_idx]
            questions = video.questions
            q_types = video.q_types
            answers = self.run(video)

            video_correct = 0
            for idx, predicted in enumerate(answers):
                actual = expected[idx]
                q_type = q_types[idx]
                if actual == predicted:
                    inc_in_map(correct, q_type)
                    video_correct += 1
                    if verbose:
                        print(f"Q{idx}: Correct.")

                else:
                    inc_in_map(incorrect, q_type)
                    question = questions[idx]
                    if verbose:
                        print(f"Q{idx}: Incorrect. Question: {question} -- "
                              f"Answer: Predicted '{predicted}', actual '{actual}'")

            acc = video_correct / len(questions)
            print(f"Video [{video_idx + 1:4}/{len(data):4}] "
                  f"-- {video_correct:2} / {len(questions):2} "
                  f"-- Accuracy: {acc:.0%}")

        return correct, incorrect

    def process(self, video):
        self.tracker.reset()
        self._time_func(self.prop_classifier.run_, (video,), "Properties")
        self._time_func(self.tracker.run_, (video,), "Tracker")
        self._time_func(self.relation_classifier.run_, (video,), "Relations")
        self._time_func(self.event_detector.run_, (video,), "Events")

    def _time_func(self, func, args, timings_key):
        start = time.time()
        result = func(*args)
        total = time.time() - start
        self._eval_timings[timings_key] += total
        return result

    def print_timings(self):
        timings = list(self._eval_timings.items())
        sorted(timings, key=lambda pair: pair[1], reverse=True)
        total = sum(list(map(lambda pair: pair[1], timings)))

        print("Timings:")
        print(f"{'Component':<15}{'Time':<15}Share")
        for component, t in timings:
            print(f"{component:<15}{t:<15.2f}{t / total:.1%}")
        print()
