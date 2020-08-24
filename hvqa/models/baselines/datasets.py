import spacy
import torch
from collections import OrderedDict
from sklearn.decomposition import PCA

from torch.utils.data import Dataset
import torchvision.transforms as T

import hvqa.util.func as util
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.util.exceptions import UnknownQuestionTypeException, UnknownAnswerException


class _AbsE2EDataset(Dataset):
    def __init__(self, spec, transform, parse_q=False):
        self.spec = spec
        self.transform = transform
        self.parse_q = parse_q
        self.nlp = spacy.load("en_core_web_md", disable=["tagger", "parser", "ner"])

        self.videos = None
        self.questions = None
        self.q_types = None
        self.answers = None

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
        frames = self.videos[v_idx]

        return frames, question, q_type, answer

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


class _AbsE2EObjDataset(_AbsE2EDataset):
    def __init__(self, spec, videos, answers, transform=None, parse_q=False):
        super(_AbsE2EObjDataset, self).__init__(spec, transform, parse_q=parse_q)

        self._tracker = ObjTracker.new(spec, err_corr=False)
        self._obj_img_size = (16, 16)
        self._obj_img_transform = T.Compose([
            T.Resize(self._obj_img_size),
            T.ToTensor(),
            T.Lambda(lambda img: img.reshape(-1).numpy())
        ])

        pca_features = 16
        self._pca = PCA(pca_features)

        print("Pre-processing end-to-end dataset...")
        self._videos_objs = self._preprocess(videos)
        pca_var = sum(self._pca.explained_variance_ratio_)
        print(f"Completed pre-processing with retained PCA variance {pca_var * 100:.2f}%.")

        q_tensors = []
        q_types_ = []
        ans_tensors = []

        for v_idx, video in enumerate(videos):
            v_qs = video.questions
            v_q_types = video.q_types
            v_ans = answers[v_idx]
            q_encs, a_encs = self._encode_qas(v_qs, v_q_types, v_ans)
            q_tensors.append(q_encs)
            q_types_.append(v_q_types)
            ans_tensors.append(a_encs)

        self.videos = self._videos_objs
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

    def _preprocess(self, videos):
        [self._tracker.run_(video) for video in videos]

        objs = []
        for video in videos:
            for frame in video.frames:
                objs.extend(frame.objs)

        objs = [self._obj_img_transform(obj.img) for obj in objs]
        obj_features = self._pca.fit_transform(objs)

        tensor_pos = self._position_in_obj_tensor()

        curr_video = 0
        curr_frame = 0
        curr_obj = 0

        videos_objs = []
        video_objs = []
        frame_objs = []
        for obj_idx, obj_feat in enumerate(obj_features):
            frame = videos[curr_video].frames[curr_frame]
            obj = frame.objs[curr_obj]
            obj_ = util.encode_obj_vector(self.spec, obj, obj_feat, tensor_pos=tensor_pos)
            frame_objs.append(obj_)
            curr_obj += 1

            # If at end of frame
            num_objs = len(frame.objs)
            if len(frame_objs) == num_objs:
                video_objs.append(frame_objs)
                frame_objs = []
                curr_obj = 0
                curr_frame += 1

            # If at end of video
            if len(video_objs) == 32:
                videos_objs.append(video_objs)
                video_objs = []
                curr_frame = 0
                curr_video += 1

        print(f"Matched objects up with {len(videos_objs)} videos.")
        return videos_objs

    @staticmethod
    def from_baseline_dataset(spec, dataset, transform=None, parse_q=False):
        print("Using VideoDataset rather than BaselineDataset...")
        return E2EObjDataset.from_video_dataset(spec, dataset, transform, parse_q=parse_q)

    @staticmethod
    def from_video_dataset(spec, dataset, transform=None, parse_q=False):
        raise NotImplementedError()

    def _position_in_obj_tensor(self):
        raise NotImplementedError()


class E2EObjDataset(_AbsE2EObjDataset):
    def __init__(self, spec, videos, answers, transform=None, parse_q=False):
        super(E2EObjDataset, self).__init__(spec, videos, answers, transform, parse_q=parse_q)

        q_tensors = []
        q_types_ = []
        ans_tensors = []

        for v_idx, video in enumerate(videos):
            v_qs = video.questions
            v_q_types = video.q_types
            v_ans = answers[v_idx]
            q_encs, a_encs = self._encode_qas(v_qs, v_q_types, v_ans)
            q_tensors.append(q_encs)
            q_types_.append(v_q_types)
            ans_tensors.append(a_encs)

        self.videos = self._videos_objs
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

    @staticmethod
    def from_video_dataset(spec, dataset, transform=None, parse_q=False):
        videos = []
        answers = []
        for v_idx in range(len(dataset)):
            video, v_ans = dataset[v_idx]
            videos.append(video)
            answers.append(v_ans)

        e2e_dataset = E2EObjDataset(spec, videos, answers, transform=transform, parse_q=parse_q)
        return e2e_dataset

    def _position_in_obj_tensor(self):
        return False


class TvqaDataset(_AbsE2EObjDataset):
    def __init__(self, spec, videos, raw_videos, answers):
        super(TvqaDataset, self).__init__(spec, videos, answers, None, parse_q=True)

        q_tensors = []
        q_types_ = []
        ans_tensors = []

        for v_idx, video in enumerate(videos):
            v_qs = video.questions
            v_q_types = video.q_types
            v_ans = answers[v_idx]
            q_encs, a_encs = self._encode_qas(v_qs, v_q_types, v_ans)
            q_tensors.append(q_encs)
            q_types_.append(v_q_types)
            ans_tensors.append(a_encs)

        self.videos = self._videos_objs
        self.raw_videos = raw_videos
        self.questions = q_tensors
        self.q_types = q_types_
        self.answers = ans_tensors

    def __len__(self):
        return len(self.questions) * 10

    def __getitem__(self, item):
        v_idx = item // 10
        q_idx = item % 10

        question = self.questions[v_idx][q_idx]
        q_type = self.q_types[v_idx][q_idx]
        answer = self.answers[v_idx][q_idx]
        frames = self.videos[v_idx]
        raw_video = self.raw_videos[v_idx]

        return (frames, raw_video), question, q_type, answer

    @staticmethod
    def from_video_dataset(spec, dataset, transform=None, parse_q=True):
        videos = []
        answers = []
        raw_videos = []
        for v_idx in range(len(dataset)):
            video, v_answers = dataset[v_idx]
            videos.append(video)
            answers.append(v_answers)
            raw_video = [frame.img for frame in video.frames]
            raw_videos.append(raw_video)

        tvqa_dataset = TvqaDataset(spec, videos, raw_videos, answers)
        return tvqa_dataset

    def _position_in_obj_tensor(self):
        return True


class BasicDataset(Dataset):
    def __init__(self, data):
        super(BasicDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
