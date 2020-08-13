import spacy
import torch
from more_itertools import grouper
from collections import OrderedDict
from torch.utils.data import Dataset

import hvqa.util.func as util
from hvqa.models.baselines.networks import PropRelNetwork, EventNetwork
from hvqa.util.exceptions import UnknownQuestionTypeException, UnknownAnswerException


class _AbsE2EDataset(Dataset):
    def __init__(self, spec, transform, parse_q=False):
        self.spec = spec
        self.transform = transform
        self.parse_q = parse_q
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform):
        raise NotImplementedError()

    def _encode_qas(self, questions, q_types, answers):
        if self.parse_q:
            parsed = self._parse_questions(questions, q_types)
            q_encs = [torch.tensor(parsed_q) for parsed_q in parsed]

        else:
            questions = [question.lower() for question in questions]
            tokens = list(self.nlp.pipe(questions))
            q_encs = [torch.tensor([q_token.vector for q_token in q_tokens]) for q_tokens in tokens]

        a_encs = self._encode_answers(q_types, answers)
        return q_encs, a_encs

    def _parse_questions(self, questions, q_types):
        parsed = []
        for idx, question in enumerate(questions):
            q_type = q_types[idx]
            parsed_q = self._parse_question(question, q_type)
            parsed.append(parsed_q)

        return parsed

    def _parse_question(self, question, q_type):
        num_values = len(self.spec.prop_values("rotation")) + len(self.spec.prop_values("colour"))
        num_obj_types = len(self.spec.obj_types())
        num_events = len(self.spec.actions) + len(self.spec.effects)
        num_frames = self.spec.num_frames

        encs = OrderedDict([
            # Property question encoding
            ("props_q_0", [0.0] * self.spec.num_props()),
            ("values_q_0", [0.0] * num_values),
            ("obj_types_q_0", [0.0] * num_obj_types),
            ("frames_q_0", [0.0] * num_frames),

            # Relation question encoding
            ("relation_q_1", [0.0] * len(self.spec.relations)),
            ("obj_types_q_1", [0.0] * num_obj_types),
            ("sec_obj_types_q_1", [0.0] * num_obj_types),
            ("values_q_1", [0.0] * num_values),
            ("sec_values_q_1", [0.0] * num_values),
            ("frames_q_1", [0.0] * num_frames),

            # Action question encoding
            ("frames_q_2", [0.0] * num_frames),

            # Changed property question encoding
            ("obj_types_q_3", [0.0] * num_obj_types),
            ("frames_q_3", [0.0] * num_frames),

            # Repetition count question encoding
            ("obj_types_q_4", [0.0] * num_obj_types),
            ("events_q_4", [0.0] * num_events),

            # Repeating action question encoding
            ("obj_types_q_5", [0.0] * num_obj_types),
            ("number_q_5", [0.0] * num_frames),

            # State transition question encoding
            ("obj_types_q_6", [0.0] * num_obj_types),
            ("events_q_6", [0.0] * num_events),
            ("occ_q_6", [0.0] * 5),

            # Explanation question encoding
            ("rotation_q_7", [0.0] * len(self.spec.prop_values("rotation"))),

            # Counter-factual question encoding
            ("obj_types_q_8", [0.0] * num_obj_types),
            ("colour_q_8", [0.0] * len(self.spec.prop_values("colour")))
        ])

        if q_type == 0:
            prop, val, cls, frame_idx = self.spec.qa.parse_prop_question(question)

            self._set_one_hot(prop, self.spec.prop_names(), encs["props_q_0"])
            self._set_prop_val_one_hot(val, encs["values_q_0"])
            self._set_one_hot(cls, self.spec.obj_types(), encs["obj_types_q_0"])
            encs["frames_q_0"][frame_idx] = 1.0

        elif q_type == 1:
            rel, obj1_cls, obj1_val, obj2_cls, obj2_val, frame_idx = self.spec.qa.parse_relation_question(question)

            self._set_one_hot(rel, self.spec.relations, encs["relation_q_1"])
            self._set_one_hot(obj1_cls, self.spec.obj_types(), encs["obj_types_q_1"])
            self._set_prop_val_one_hot(obj1_val, encs["values_q_1"])
            self._set_one_hot(obj2_cls, self.spec.obj_types(), encs["sec_obj_types_q_1"])
            self._set_prop_val_one_hot(obj2_val, encs["sec_values_q_1"])
            encs["frames_q_1"][frame_idx] = 1.0

        elif q_type == 2:
            frame_idx = self.spec.qa.parse_event_question(question)
            encs["frames_q_2"][frame_idx] = 1.0

        elif q_type == 3:
            _, cls, frame_idx = self.spec.qa.parse_q_3(question)
            self._set_one_hot(cls, self.spec.obj_types(), encs["obj_types_q_3"])
            encs["frames_q_3"][frame_idx] = 1.0

        elif q_type == 4:
            _, cls, event = self.spec.qa.parse_q_4(question)
            self._set_one_hot(cls, self.spec.obj_types(), encs["obj_types_q_4"])
            events = self.spec.actions + self.spec.effects
            self._set_one_hot(self.spec.from_internal("event", event), events, encs["events_q_4"])

        elif q_type == 5:
            num, _, cls = self.spec.qa.parse_q_5(question)
            self._set_one_hot(cls, self.spec.obj_types(), encs["obj_types_q_5"])
            encs["number_q_5"][num] = 1.0

        elif q_type == 6:
            _, cls, occ, event = self.spec.qa.parse_q_6(question)
            self._set_one_hot(cls, self.spec.obj_types(), encs["obj_types_q_6"])
            events = self.spec.actions + self.spec.effects
            self._set_one_hot(self.spec.from_internal("event", event), events, encs["events_q_6"])
            encs["occ_q_6"][occ - 1] = 1.0

        elif q_type == 7:
            rot = self.spec.qa.parse_explanation_question(question)
            self._set_one_hot(rot, self.spec.prop_values("rotation"), encs["rotation_q_7"])

        elif q_type == 8:
            colour = self.spec.qa.parse_counterfactual_question(question)
            self._set_one_hot(colour, self.spec.prop_values("colour"), encs["colour_q_8"])
            self._set_one_hot("octopus", self.spec.obj_types(), encs["obj_types_q_8"])

        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        question_enc = []
        for _, enc in encs.items():
            question_enc.extend(enc)

        return question_enc

    def _set_one_hot(self, value, values, enc):
        val_idx = values.index(value)
        enc[val_idx] = 1.0

    def _set_prop_val_one_hot(self, val, enc):
        if val is not None:
            prop = self.spec.find_prop(val)
            val_idx = 0 if prop == "colour" else len(self.spec.prop_values("colour"))
            val_idx += self.spec.prop_values(prop).index(val)
            enc[val_idx] = 1.0

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


class E2EDataset(_AbsE2EDataset):
    def __init__(self, spec, frames, questions, q_types, answers, transform, lang_only=False):
        super(E2EDataset, self).__init__(spec, transform)

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
        :return: frames (could be None), question, q_type, answer
        """

        v_idx = item // 10
        q_idx = item % 10

        question = self.questions[v_idx][q_idx]
        q_type = self.q_types[v_idx][q_idx]
        answer = self.answers[v_idx][q_idx]

        frames = None
        if not self.lang_only:
            frames = self.frames[v_idx]
            if self.transform is not None:
                frames = [self.transform(frame) for frame in frames]

        return frames, question, q_type, answer

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform=None, lang_only=False):
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

        e2e_dataset = E2EDataset(spec, frames, questions, q_types, answers, transform, lang_only=lang_only)
        return e2e_dataset


class E2EFilterDataset(_AbsE2EDataset):
    def __init__(self, spec, frames, questions, q_types, answers, transform):
        super(E2EFilterDataset, self).__init__(spec, transform)

        q_encs, a_encs = self._encode_qas(questions, q_types, answers)

        self.frames = frames
        self.questions = q_encs
        self.q_types = q_types
        self.answers = a_encs

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        frame = self.frames[item]
        question = self.questions[item]
        q_type = self.q_types[item]
        answer = self.answers[item]
        return frame, question, q_type, answer

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform, filter_qs=None):
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
                    frame = E2EFilterDataset._encode_frame(spec, v_frames, question, q_type, transform)
                    frames.append(frame)
                    questions.append(question)
                    q_types.append(q_type)
                    answers.append(answer)

        e2e_dataset = E2EFilterDataset(spec, frames, questions, q_types, answers, transform)
        return e2e_dataset

    @staticmethod
    def _encode_frame(spec, frames, question, q_type, transform):
        if q_type == 0:
            _, _, _, frame_idx = spec.qa.parse_prop_question(question)
        elif q_type == 1:
            _, _, _, _, _, frame_idx = spec.qa.parse_relation_question(question)
        elif q_type == 2:
            frame_idx = spec.qa.parse_event_question(question)
        else:
            raise UnknownQuestionTypeException(f"Filter questions must be of type: 0, 1 or 2")

        frame = frames[frame_idx]
        if q_type == 2:
            next_frame = frames[frame_idx + 1]
            frame = transform(frame)
            next_frame = transform(next_frame)
            frames_tensor = torch.cat((frame, next_frame), dim=0)
        else:
            frames_tensor = transform(frame)

        return frames_tensor


class E2EPreDataset(_AbsE2EDataset):
    def __init__(self, spec, frames, questions, q_types, answers, transform=None, collate="stack", parse_q=False):
        super(E2EPreDataset, self).__init__(spec, transform, parse_q=parse_q)

        frame_feat_extr, event_feat_extr = self._load_feat_extr(spec)

        self._device = util.get_device()

        self._frame_feat_extr = frame_feat_extr.to(self._device)
        for param in self._frame_feat_extr.parameters():
            param.requires_grad = False

        self._event_feat_extr = event_feat_extr.to(self._device)
        for param in self._event_feat_extr.parameters():
            param.requires_grad = False

        self.collate = collate

        print("Pre-processing end-to-end dataset...")
        frames = self._preprocess(frames, questions, q_types)
        print("Completed pre-processing.")

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

        self.frames = frames
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
        :return: frames, question, q_type, answer
        """

        v_idx = item // 10
        q_idx = item % 10

        question = self.questions[v_idx][q_idx]
        q_type = self.q_types[v_idx][q_idx]
        answer = self.answers[v_idx][q_idx]
        frames = self.frames[v_idx]

        return frames, question, q_type, answer

    def _preprocess(self, frames, question, q_types):
        group_videos = 16
        frame_feats = self._extr_frame_feats(frames, group_videos)
        event_feats = self._extr_event_feats(frames, group_videos)

        feats = []
        if self.collate == "stack":
            for idx, frame_feat in enumerate(frame_feats):
                event_feat = event_feats[idx].reshape(-1)
                frame_feat = frame_feat.reshape(-1)
                feat = torch.cat((frame_feat, event_feat), dim=0)
                feats.append(feat)

        return feats

    def _extr_frame_feats(self, frames, group_videos):
        videos_feats = []
        for videos in grouper(frames, group_videos):
            videos = [video for video in videos if video is not None]
            frames_ = [self.transform(frame) for v_frames in videos for frame in v_frames]

            frames_ = torch.stack(frames_).to(self._device)
            v_feats = self._frame_feat_extr(frames_)

            for v_idx in range(len(videos)):
                start = 32 * v_idx
                end = 32 * (v_idx + 1)
                video_feats = v_feats[start:end, :]
                videos_feats.append(video_feats)

        return videos_feats

    def _extr_event_feats(self, frames, group_videos):
        events_feats = []
        for videos in grouper(frames, group_videos):
            videos = [video for video in videos if video is not None]
            frames_ = [[self.transform(frame) for frame in v_frames] for v_frames in videos]
            frame_pairs = [zip(v_frames, v_frames[1:]) for v_frames in frames_]
            frame_pairs = [torch.cat((im1, im2), dim=0) for v_frames in frame_pairs for im1, im2 in v_frames]

            frame_pairs_ = torch.stack(frame_pairs).to(self._device)
            feats = self._event_feat_extr(frame_pairs_)

            for v_idx in range(len(videos)):
                start = 31 * v_idx
                end = 31 * (v_idx + 1)
                event_feats = feats[start:end, :]
                events_feats.append(event_feats)

        return events_feats

    @staticmethod
    def _load_feat_extr(spec):
        prop_rel = util.load_model(PropRelNetwork, "saved-models/pre/prop-rel/network.pt", spec)
        event = util.load_model(EventNetwork, "saved-models/pre/event/network.pt", spec)
        frame_feat_extr = prop_rel.feat_extr
        event_feat_extr = event.feat_extr
        return frame_feat_extr, event_feat_extr

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform=None, collate="stack", parse_q=False):
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

        e2e_dataset = E2EPreDataset(spec, frames, questions, q_types, answers,
                                    transform=transform, collate=collate, parse_q=parse_q)
        return e2e_dataset


class E2EObjDataset(_AbsE2EDataset):
    def __init__(self, spec, frames, questions, q_types, answers, transform=None, parse_q=False):
        super(E2EObjDataset, self).__init__(spec, transform, parse_q=parse_q)

        print("Pre-processing end-to-end dataset...")
        frames = self._preprocess(frames, questions, q_types)
        print("Completed pre-processing.")

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

        self.frames = frames
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

    def _preprocess(self, frames, questions, q_types):
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
        :return: frames, question, q_type, answer
        """

        v_idx = item // 10
        q_idx = item % 10

        question = self.questions[v_idx][q_idx]
        q_type = self.q_types[v_idx][q_idx]
        answer = self.answers[v_idx][q_idx]
        frames = self.frames[v_idx]

        return frames, question, q_type, answer

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform, parse_q=False):
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

        e2e_dataset = E2EObjDataset(spec, frames, questions, q_types, answers, transform=transform, parse_q=parse_q)
        return e2e_dataset
