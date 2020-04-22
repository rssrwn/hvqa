# Abstract QA Model Class File

import time

from hvqa.util.interfaces import Model
from hvqa.util.func import inc_in_map


class _AbsModel(Model):
    """
    Abstract QA Model Class
    Most of the models will run the same pipeline which is contained in this class
    Models need to set the components within their __init__ functions
    Models can also override methods if required
    """

    def __init__(self, detector, properties, tracker, relations, events, qa):
        self.obj_detector = detector
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

        self._timings = {
            "Detector": 0,
            "Properties": 0,
            "Tracker": 0,
            "Relations": 0,
            "Events": 0,
            "QA": 0
        }

    def run(self, frames, questions, q_types):
        """
        Generate answers to given questions

        :param frames: List of PIL Images
        :param questions: Questions about video: [str]
        :param q_types: Type of each question: [int]
        :return: Answers: [str]
        """

        video = self.process(frames)
        video.questions = questions
        video.q_types = q_types
        self._time_func(self.qa_system.run_, (video,), "QA")
        answers = video.answers
        return answers

    def train(self, data, verbose=True):
        print("Training VideoQA model...")

        for component in self.components:
            component.train(data, verbose)

        print("Completed VideoQA model training")

    @staticmethod
    def load(path):
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
            frames, video_dict = data[video_idx]
            questions = video_dict["questions"]
            q_types = video_dict["question_types"]

            answers = self.run(frames, questions, q_types)
            expected = video_dict["answers"]

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

    def process(self, frames):
        video = self.extract_objs(frames)

        self.tracker.reset()
        self._time_func(self.prop_classifier.run_, (video,), "Properties")
        self._time_func(self.tracker.run_, (video,), "Tracker")
        self._time_func(self.relation_classifier.run_, (video,), "Relations")
        self._time_func(self.event_detector.run_, (video,), "Events")

        return video

    def extract_objs(self, frames):
        """
        Builds structured knowledge from video frames
        Extracts bboxs and class for each object in each frame

        :param frames: List of PIL frames
        """

        video = self._time_func(self.obj_detector.detect_objs, (frames,), "Detector")
        return video

    def _time_func(self, func, args, timings_key):
        start = time.time()
        result = func(*args)
        total = time.time() - start
        self._timings[timings_key] += total
        return result

    def print_timings(self):
        timings = list(self._timings.items())
        sorted(timings, key=lambda pair: pair[1], reverse=True)
        total = sum(list(map(lambda pair: pair[1], timings)))

        print("Timings:")
        print(f"{'Component':<15}{'Time':<15}Share")
        for component, t in timings:
            print(f"{component:<15}{t:<15.2f}{t / total:.1%}")
        print()
