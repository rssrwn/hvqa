import spacy
import torch
from torch.utils.data import Dataset

import hvqa.util.func as util
from hvqa.util.exceptions import UnknownQuestionTypeException


class EndToEndDataset(Dataset):
    def __init__(self, spec, frames, questions, q_types, answers, lang_only=False):
        self.spec = spec
        self.lang_only = lang_only
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

        frame_tensors = []
        q_tensors = []
        ans_tensors = []

        for v_idx, v_frames in enumerate(frames):
            v_qs = questions[v_idx]
            v_q_types = q_types[v_idx]
            v_ans = answers[v_idx]
            q_encs, a_encs = self._encode_qas(v_qs, v_q_types, v_ans)
            q_tensors.append(q_encs)
            ans_tensors.append(a_encs)

            if not lang_only:
                frame_encs = self._encode_frames(v_frames)
                frame_tensors.append(frame_encs)

        self.frames = frame_tensors
        self.questions = q_tensors
        self.answers = ans_tensors

    def _encode_qas(self, questions, q_types, answers):
        questions = [question.lower() for question in questions]
        tokens = list(self.nlp.pipe(questions))
        q_encs = [token.vector for token in tokens]
        a_encs = self._encode_answers(q_types, answers)
        q_encs = torch.tensor(q_encs)
        a_encs = torch.tensor(a_encs)
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
            ans_enc = util.property_encoding(self.spec, prop, answer)
            if prop == "colour":
                ans_enc.append([0.0] * 4)
            else:
                ans_enc = ([0.0] * 7) + ans_enc

        elif q_type == 1:
            ans_enc = 1.0 if answer == "yes" else 0.0

        elif q_type == 2:
            actions = self.spec.actions
            ans_enc = list(map(lambda a: 1.0 if a == answer else 0.0, actions))

        elif q_type == 3:
            prop, before, after = self.spec.qa.parse_ans_3(answer)
            vals = self.spec.prop_values(prop)
            before_idx = vals.index(before)
            after_idx = vals.index(after)

            col_enc = [0.0] * (7 * 7)
            rot_enc = [0.0] * (4 * 4)
            if prop == "colour":
                idx = (before_idx * 7) + after_idx
                col_enc[idx] = 1.0
            elif prop == "rotation":
                idx = (before_idx * 4) + after_idx
                rot_enc[idx] = 1.0

            ans_enc = col_enc + rot_enc

        elif q_type == 4:
            ans_enc = [0.0] * 31
            answer = int(answer)
            ans_enc[answer] = 1.0

        elif q_type == 5:
            events = self.spec.actions + self.spec.effects
            ans_enc = list(map(lambda e: 1.0 if e == answer else 0.0, events))

        elif q_type == 6:
            actions = self.spec.actions
            ans_enc = list(map(lambda a: 1.0 if a == answer else 0.0, actions))

        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans_enc

    def _encode_frames(self, frames):
        pass

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, item):
        questions = self.questions[item]
        answers = self.answers[item]

        frames = None
        if not self.lang_only:
            frames = self.frames[item]

        return frames, questions, answers

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
