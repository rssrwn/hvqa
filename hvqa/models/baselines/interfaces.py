from hvqa.util.interfaces import BaselineModel


NUM_Q_TYPES = 9


class _AbsBaselineModel(BaselineModel):
    def __init__(self, spec):
        self.spec = spec
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
        """
        Eval result the video, print and update results dicts

        :param v_idx: Video index
        :param n_vs: Number of videos
        :param results: List of tuples (question, q_type, predicted, answer)
        :param verbose: Additional printing
        :return: Number of answers correct in this video
        """

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

            self._q_type_total[q_type] += 1

        self._correct += video_correct
        self._total += len(results)

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
            incorrect_q_t = total_q_t - correct_q_t
            acc = correct_q_t / total_q_t if total_q_t > 0 else 0.0
            print(f"{q_type:<20}{correct_q_t:<15}{incorrect_q_t:<15}{acc:.1%}")

        acc = self._correct / self._total
        print(f"\nNum correct: {self._correct}")
        print(f"Total: {self._total}")
        print(f"Accuracy: {acc:.1%}\n")

        self._reset_results()
