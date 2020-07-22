import random

from hvqa.util.interfaces import BaselineModel
from hvqa.util.exceptions import UnknownQuestionTypeException


NUM_Q_TYPES = 7


class RandomAnsModel(BaselineModel):
    def __init__(self, spec):
        self.spec = spec

    def train(self, train_data, eval_data, verbose=True):
        pass

    def eval(self, eval_data, verbose=True):
        q_t_correct = dict(enumerate([0] * NUM_Q_TYPES))
        q_t_total = dict(enumerate([0] * NUM_Q_TYPES))
        correct = 0
        total = 0

        num_vs = len(eval_data)
        for idx in range(num_vs):
            _, qs, q_types, answers = eval_data[idx]
            v_correct = self._eval_video(idx, num_vs, qs, q_types, answers, q_t_correct, q_t_total, verbose)
            correct += v_correct
            total += len(qs)

        print("\nResults:")
        print(f"{'Question Type':<20}{'Correct':<15}{'Incorrect':<15}Accuracy")
        for q_type in range(NUM_Q_TYPES):
            correct_q_t = q_t_correct[q_type]
            total_q_t = q_t_total[q_type]
            incorrect_q_t = total - correct_q_t
            acc = correct_q_t / total_q_t
            print(f"{q_type:<20}{correct_q_t:<15}{incorrect_q_t:<15}{acc:.1%}")

        acc = correct / total
        print(f"\nNum correct: {correct}")
        print(f"Total: {total}")
        print(f"Accuracy: {acc:.1%}\n")

    def _eval_video(self, v_idx, n_vs, questions, q_types, answers, q_type_correct, q_type_total, verbose):
        video_correct = 0

        for idx, question in enumerate(questions):
            q_type = q_types[idx]
            answer = answers[idx]
            predicted = self._answer_question(question, q_type)
            if predicted == answer:
                q_type_correct[q_type] += 1
                video_correct += 1
                if verbose:
                    print(f"Q{idx}: Correct.")

            else:
                if verbose:
                    print(f"Q{idx}: Incorrect. Question {question} -- "
                          f"Answer: Predicted '{predicted}', actual '{answer}'")

            q_type_total[q_type] += 1

        acc = video_correct / len(questions)
        print(f"Video [{v_idx + 1:4}/{n_vs:4}] "
              f"-- {video_correct:2} / {len(questions):2} "
              f"-- Accuracy: {acc:.0%}")

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
