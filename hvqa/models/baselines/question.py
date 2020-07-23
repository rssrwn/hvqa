import random

from hvqa.util.interfaces import BaselineModel
from hvqa.util.exceptions import UnknownQuestionTypeException


NUM_Q_TYPES = 7


class _AbsBaselineModel(BaselineModel):
    def __init__(self):
        self._reset_results()

    def _reset_results(self):
        self._correct = 0
        self._total = 0
        self._q_type_correct = dict(enumerate([0] * NUM_Q_TYPES))
        self._q_type_total = dict(enumerate([0] * NUM_Q_TYPES))

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError()

    def eval(self, eval_data, verbose=True):
        raise NotImplementedError()

    @staticmethod
    def load(spec, path):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def _eval_video_results(self, v_idx, n_vs, results, verbose):
        video_correct = 0
        for q_idx, result in enumerate(results):
            question, q_type, predicted, answer = result
            if predicted == answer:
                self._q_type_correct[q_type] += 1
                video_correct += 1
                if verbose:
                    print(f"Q{q_idx}: Correct.")

            else:
                if verbose:
                    print(f"Q{q_idx}: Incorrect. Question {question} -- "
                          f"Answer: Predicted '{predicted}', actual '{answer}'")

        acc = video_correct / len(results)
        print(f"Video [{v_idx + 1:4}/{n_vs:4}] "
              f"-- {video_correct:2} / {len(results):2} "
              f"-- Accuracy: {acc:.0%}")

        return video_correct

    def _print_results(self):
        print("\nResults:")
        print(f"{'Question Type':<20}{'Correct':<15}{'Incorrect':<15}Accuracy")
        for q_type in range(NUM_Q_TYPES):
            correct_q_t = self._q_type_correct[q_type]
            total_q_t = self._q_type_total[q_type]
            incorrect_q_t = self._total - correct_q_t
            acc = correct_q_t / total_q_t
            print(f"{q_type:<20}{correct_q_t:<15}{incorrect_q_t:<15}{acc:.1%}")

        acc = self._correct / self._total
        print(f"\nNum correct: {self._correct}")
        print(f"Total: {self._total}")
        print(f"Accuracy: {acc:.1%}\n")


class RandomAnsModel(_AbsBaselineModel):
    def __init__(self, spec):
        self.spec = spec
        super(RandomAnsModel, self).__init__()

    def train(self, train_data, eval_data, verbose=True):
        pass

    def eval(self, eval_data, verbose=True):
        num_vs = len(eval_data)
        for idx in range(num_vs):
            _, qs, q_types, answers = eval_data[idx]
            v_correct = self._eval_video(idx, num_vs, qs, q_types, answers, verbose)
            self._correct += v_correct
            self._total += len(qs)

        self._print_results()
        self._reset_results()

    def _eval_video(self, v_idx, n_vs, questions, q_types, answers, verbose):
        results = []
        for idx, question in enumerate(questions):
            q_type = q_types[idx]
            answer = answers[idx]
            predicted = self._answer_question(question, q_type)
            results.append((question, q_type, predicted, answer))

        video_correct = self._eval_video_results(v_idx, n_vs, results, verbose)
        return video_correct

    def _answer_question(self, question, q_type):
        if q_type == 0:
            prop, _, _, _ = self.spec.qa.parse_prop_question(question)
            ans = random.choice(self.spec.prop_values(prop))
        elif q_type == 1:
            ans = random.choice(["yes", "no"])
        elif q_type == 2:
            ans = random.choice(self.spec.actions)
        elif q_type == 3:
            prop = random.choice(self.spec.prop_names())
            from_val = random.choice(self.spec.prop_values(prop))
            to_val = random.choice(self.spec.prop_values(prop))
            ans = f"Its {prop} changed from {from_val} to {to_val}"
        elif q_type == 4:
            ints = list(range(self.spec.num_frames - 1))
            ans = str(random.choice(ints))
        elif q_type == 5:
            ans = random.choice(self.spec.actions + self.spec.effects)
        elif q_type == 6:
            ans = random.choice(self.spec.actions)
        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans

    @staticmethod
    def load(spec, path):
        pass

    def save(self, path):
        pass
