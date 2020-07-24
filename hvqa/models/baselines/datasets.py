import spacy
from torch.utils.data import Dataset


class EndToEndDataset(Dataset):
    def __init__(self, spec, frames, questions, q_types, answers, lang_only=False):
        self.spec = spec
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
        return q_encs, a_encs

    def _encode_answers(self, q_types, answers):
        pass

    def _encode_frames(self, frames):
        pass

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

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
