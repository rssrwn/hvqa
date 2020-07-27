import spacy
import torch
from torch.utils.data import Dataset

from hvqa.util.exceptions import UnknownQuestionTypeException


class EndToEndDataset(Dataset):
    def __init__(self, spec, frames, questions, q_types, answers, lang_only=False):
        self.spec = spec
        self.lang_only = lang_only
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

        frame_tensors = []
        q_tensors = []
        q_types_ = []
        ans_tensors = []

        for v_idx, v_frames in enumerate(frames):
            v_qs = questions[v_idx]
            v_q_types = q_types[v_idx]
            v_ans = answers[v_idx]
            q_encs, a_encs = self._encode_qas(v_qs, v_q_types, v_ans)
            q_tensors.append(q_encs)
            q_types_.append(v_q_types)
            ans_tensors.append(a_encs)

            if not lang_only:
                frame_encs = self._encode_frames(v_frames)
                frame_tensors.append(frame_encs)

        self.frames = frame_tensors
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

    def _encode_qas(self, questions, q_types, answers):
        questions = [question.lower() for question in questions]
        tokens = list(self.nlp.pipe(questions))
        q_encs = [torch.tensor([q_token.vector for q_token in q_tokens]) for q_tokens in tokens]
        a_encs = self._encode_answers(q_types, answers)
        return q_encs, a_encs

    def _encode_answers(self, q_types, answers):
        ans_encs = []
        for idx, ans in enumerate(answers):
            q_type = q_types[idx]
            ans_enc = self._encode_answer(ans, q_type)
            ans_enc = torch.tensor(ans_enc)
            ans_encs.append(ans_enc)

        return ans_encs

    def _encode_answer(self, answer, q_type):
        if q_type == 0:
            prop = self.spec.find_prop(answer)
            ans_enc = self.spec.to_internal(prop, answer)
            if prop == "rotation":
                ans_enc += 7

        elif q_type == 1:
            ans_enc = 1 if answer == "yes" else 0

        elif q_type == 2:
            actions = self.spec.actions
            ans_enc = actions.index(answer)

        elif q_type == 3:
            prop, before, after = self.spec.qa.parse_ans_3(answer)
            vals = self.spec.prop_values(prop)
            before_idx = vals.index(before)
            after_idx = vals.index(after)

            ans_enc = (before_idx * 7) + after_idx
            if prop == "rotation":
                ans_enc = (7 * 7) + (before_idx * 4) + after_idx

        elif q_type == 4:
            ans_enc = int(answer)

        elif q_type == 5:
            events = self.spec.actions + self.spec.effects
            ans_enc = events.index(answer)

        elif q_type == 6:
            actions = self.spec.actions
            ans_enc = actions.index(answer)

        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans_enc

    def _encode_frames(self, frames):
        pass

    def __len__(self):
        """
        Returns number of videos multiplied by the number of questions per video

        :return: Length of dataset
        """

        return len(self.questions) * 10

    def __getitem__(self, item):
        """
        Get an item in the dataset
        An item is a video, a question and an answer
        Each question in a video is treated as a separate item

        :param item: Index into dataset
        :return: frames (could be None), questions, answers
        """

        v_idx = item // 10
        q_idx = item % 10

        question = self.questions[v_idx][q_idx]
        q_type = self.q_types[v_idx][q_idx]
        answer = self.answers[v_idx][q_idx]

        if not self.lang_only:
            frames = self.frames[v_idx]
            return frames, question, q_type, answer

        return question, q_type, answer

    @staticmethod
    def from_baseline_dataset(spec, dataset, lang_only=False):
        frames = []
        questions = []
        q_types = []
        answers = []
        for v_idx in range(len(dataset)):
            v_frames, v_qs, v_types, v_ans = dataset[v_idx]
            frames.append(v_frames)
            questions.append(v_qs)
            q_types.append(v_types)
            answers.append(v_ans)

        e2e_dataset = EndToEndDataset(spec, frames, questions, q_types, answers, lang_only=lang_only)
        return e2e_dataset
