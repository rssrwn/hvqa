import spacy
import torch
from torch.utils.data import Dataset

from hvqa.util.exceptions import UnknownQuestionTypeException, UnknownAnswerException


class _AbsEndToEndDataset(Dataset):
    def __init__(self, spec, transform=None):
        self.spec = spec
        self.transform = transform
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    def from_baseline_dataset(self, spec, dataset, transform=None):
        raise NotImplementedError()

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

        elif q_type == 7:
            if answer == "The octopus ate a bag":
                ans_enc = 0
            elif answer == "The fish was eaten":
                ans_enc = 1
            elif answer == "The bag was eaten":
                ans_enc = 2
            else:
                raise UnknownAnswerException(f"Answer {answer} unknown for explanation questions")

        elif q_type == 8:
            colours = self.spec.prop_values("colour")
            ans_enc = colours.index(answer)

        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans_enc


class EndToEndDataset(_AbsEndToEndDataset):
    def __init__(self, spec, frames, questions, q_types, answers, lang_only=False, transform=None):
        super(EndToEndDataset, self).__init__(spec, transform=transform)

        self.lang_only = lang_only

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
                frame_tensors.append(v_frames)

        self.frames = frame_tensors
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

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
            if self.transform is not None:
                frames = [self.transform(frame) for frame in frames]

            return frames, question, q_type, answer

        return question, q_type, answer

    @staticmethod
    def from_baseline_dataset(spec, dataset, lang_only=False, transform=None):
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

        e2e_dataset = EndToEndDataset(spec, frames, questions, q_types, answers,
                                      lang_only=lang_only, transform=transform)
        return e2e_dataset


class EndToEndPreTrainDataset(_AbsEndToEndDataset):
    def __init__(self, spec, frames, questions, q_types, answers, transform=None):
        super(EndToEndPreTrainDataset, self).__init__(spec, transform=transform)

        self.frames = frames
        self.questions = questions
        self.q_types = q_types
        self.answers = answers

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def from_baseline_dataset(self, spec, dataset, transform=None, filter_qs=None):
        """
        Create dataset from baseline dataset

        :param spec: Environment specification
        :param dataset: BaselineDataset
        :param transform: Image transform to apply to each frame
        :param filter_qs: Keep questions of this q_type (only applies to q_types 0, 1 and 2)
        :return: EndToEndPreTrainDataset
        """

        frames = []
        questions = []
        q_types = []
        answers = []

        allowed_q_types = {0, 1, 2}
        filter_qs = [] if filter_qs is None else filter_qs
        q_filter = set([q_type for q_type in filter_qs if q_type in allowed_q_types])
        print(f"Keeping following question types: {q_filter}")

        for v_idx in range(len(dataset)):
            v_frames, v_qs, v_types, v_ans = dataset[v_idx]
            for q_idx, q_type in enumerate(v_types):
                question = v_qs[q_idx]
                answer = v_ans[q_idx]
                if q_type in q_filter:
                    frame = self._encode_frame(v_frames, question, q_type)
                    frames.append(frame)
                    questions.append(question)
                    q_types.append(q_type)
                    answers.append(answer)

        e2e_dataset = EndToEndDataset(spec, frames, questions, q_types, answers, transform=transform)
        return e2e_dataset

    def _encode_frame(self, frames, question, q_type):
        pass
