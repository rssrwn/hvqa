import pickle
import random
from pathlib import Path

from hvqa.util.exceptions import UnknownQuestionTypeException
from hvqa.models.baselines.interfaces import _AbsBaselineModel


class RandomAnsModel(_AbsBaselineModel):
    def __init__(self, spec):
        super(RandomAnsModel, self).__init__(spec)

    def train(self, train_data, eval_data, verbose=True):
        pass

    def eval(self, eval_data, verbose=True):
        num_vs = len(eval_data)
        for idx in range(num_vs):
            _, qs, q_types, answers = eval_data[idx]
            v_correct = self._eval_video(idx, num_vs, qs, q_types, answers, verbose)

        self._print_results()

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
        elif q_type == 7:
            answers = ["The octopus ate a bag", "The fish was eaten", "The bag was eaten"]
            ans = random.choice(answers)
        elif q_type == 8:
            colours = self.spec.prop_values("colour")
            ans = random.choice(colours)
        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans

    @staticmethod
    def load(spec, path):
        pass

    def save(self, path):
        pass


class BestChoiceModel(_AbsBaselineModel):
    def __init__(self, spec, answers=None):
        self._answers = answers
        super(BestChoiceModel, self).__init__(spec)

    def train(self, train_data, eval_data, verbose=True):
        counts = {}

        print("Training best choice language-only model...")

        num_vs = len(train_data)
        for idx in range(num_vs):
            _, qs, q_types, answers = train_data[idx]

            for q_idx, question in enumerate(qs):
                q_type = q_types[q_idx]
                answer = answers[q_idx]
                self._add_ans_count(counts, question, q_type, answer)

        answers = {}
        for q_type, ans_counts in counts.items():
            if q_type == 0:
                prop_answers = {}
                for prop, val_counts in ans_counts.items():
                    val, _ = max(val_counts.items(), key=lambda val_cnt: val_cnt[1])
                    prop_answers[prop] = val

                answers[q_type] = prop_answers

            else:
                ans, _ = max(ans_counts.items(), key=lambda ans_cnt: ans_cnt[1])
                answers[q_type] = ans

        self._answers = answers
        print("Completed training best-choice model.")

    def _add_ans_count(self, counts, question, q_type, answer):
        q_type_counts = counts.get(q_type)
        if q_type_counts is None:
            counts[q_type] = {}

        if q_type == 0:
            prop, _, _, _ = self.spec.qa.parse_prop_question(question)
            prop_counts = counts[q_type].get(prop)
            if prop_counts is None:
                counts[q_type][prop] = {}

            ans_count = counts[q_type][prop].get(answer)
            if ans_count is None:
                counts[q_type][prop][answer] = 0

            counts[q_type][prop][answer] += 1

        else:
            ans_count = counts[q_type].get(answer)
            if ans_count is None:
                counts[q_type][answer] = 0

            counts[q_type][answer] += 1

    def eval(self, eval_data, verbose=True):
        num_vs = len(eval_data)
        for idx in range(num_vs):
            _, qs, q_types, answers = eval_data[idx]
            v_correct = self._eval_video(idx, num_vs, qs, q_types, answers, verbose)

        self._print_results()

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
            ans = self._answers[q_type][prop]
        else:
            ans = self._answers[q_type]

        return ans

    @staticmethod
    def load(spec, path):
        load_path = Path(path)
        answers_path = load_path / "answers.json"
        with answers_path.open("rb") as f:
            pickle_text = f.read()

        answers = pickle.loads(pickle_text)
        model = BestChoiceModel(spec, answers=answers)
        return model

    def save(self, path):
        assert self._answers is not None, "Model has not yet been trained"
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        answers_path = save_path / "answers.json"
        pickle_text = pickle.dumps(self._answers)
        with answers_path.open("wb") as f:
            f.write(pickle_text)
