import random

import torch
from torch.utils.data import Dataset


DEFAULT_ITEMS_PER_REL = 50000


class _AbsRelationDataset(Dataset):
    def __init__(self, spec, videos, answers=None, items_per_rel=DEFAULT_ITEMS_PER_REL):
        """
        Create a RelationDataset

        :param spec: EnvSpec object
        :param videos: List of Video object to construct dataset from
        :param answers: Answers to videos (None if using dataset to run the component)
        :param items_per_rel: Number of items in each relation in the data returned
        """

        self.spec = spec
        self.items_per_rel = items_per_rel
        self.rel_data_map = self._collect_data(videos, answers)

    def __len__(self):
        return self.items_per_rel

    def __getitem__(self, item):
        rel_data = {rel: self.rel_data_map[rel][item] for rel in self.spec.relations}
        return rel_data

    def _collect_data(self, videos, answers):
        """
        Collect data from videos and answers

        :param videos: Videos to construct dataset from
        :param answers: Answers to videos (None if using data for running the component)
        :return: Dict from rel to data
        """

        raise NotImplementedError()

    def _match_obj(self, obj_cls, obj_val, objs):
        obj_prop = self.spec.find_prop(obj_val) if obj_val is not None else None
        for obj_ in objs:
            if obj_.cls == obj_cls and obj_val is None:
                return obj_
            elif obj_.cls == obj_cls and obj_.prop_vals[obj_prop] == obj_val:
                return obj_

        return None

    def _obj_encoding(self, obj):
        obj_enc = list(map(lambda v: 1.0 if v == obj.cls else 0.0, self.spec.obj_types()))
        obj_position = list(map(lambda p: p / 255, obj.pos))
        obj_enc.extend(obj_position)
        for prop, val in obj.prop_vals.items():
            prop_enc = self._property_encoding(prop, val)
            obj_enc.extend(prop_enc)

        return obj_enc

    def _property_encoding(self, prop, val):
        vals = self.spec.prop_values(prop)
        one_hot = list(map(lambda v: 1.0 if v == val else 0.0, vals))
        assert sum(one_hot) == 1.0, f"Val {val} is not in property values {vals}"
        return one_hot


class QARelationDataset(_AbsRelationDataset):
    def __init__(self, spec, videos, answers, items_per_rel=DEFAULT_ITEMS_PER_REL):
        assert answers is not None
        super(QARelationDataset, self).__init__(spec, videos, answers, items_per_rel)

    def _collect_data(self, videos, answers):
        rel_data_map = {rel: [] for rel in self.spec.relations}
        for video_idx, video in enumerate(videos):
            q_idxs = [idx for idx, q_type in enumerate(video.q_types) if q_type == self.spec.qa.relation_q]
            for q_idx in q_idxs:
                question = video.questions[q_idx]
                answer = answers[video_idx][q_idx]
                parsed_q = self.spec.qa.parse_relation_question(question)
                rel, obj1_cls, obj1_val, obj2_cls, obj2_val, frame_idx = parsed_q

                yes_no = self.spec.qa.parse_ans_1(answer)
                yes_no = 1.0 if yes_no == "yes" else 0.0

                obj1 = self._match_obj(obj1_cls, obj1_val, video.frames[frame_idx].objs)
                obj2 = self._match_obj(obj2_cls, obj2_val, video.frames[frame_idx].objs)

                assert obj1 is not None, f"Cannot find {obj1_val} {obj1_cls} in frame {frame_idx}"
                assert obj2 is not None, f"Cannot find {obj2_val} {obj2_cls} in frame {frame_idx}"

                obj_enc = self._obj_encoding(obj1)
                obj2_enc = self._obj_encoding(obj2)
                obj_enc.extend(obj2_enc)

                objs = torch.tensor(obj_enc)
                classification = torch.tensor([yes_no])
                rel_data_map[rel].append((objs, classification))

        sampled_rel_data_map = {}
        for rel, data in rel_data_map.items():
            sampled = random.choices(data, k=self.items_per_rel)
            sampled_rel_data_map[rel] = sampled

        return sampled_rel_data_map

    @staticmethod
    def from_video_dataset(spec, dataset, items_per_rel=DEFAULT_ITEMS_PER_REL):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        relation_data = QARelationDataset(spec, videos, answers, items_per_rel=items_per_rel)
        return relation_data


class HardcodedRelationDataset(_AbsRelationDataset):
    def __init__(self, spec, videos, answers, items_per_rel=DEFAULT_ITEMS_PER_REL):
        self.close_to_def = 5
        super(HardcodedRelationDataset, self).__init__(spec, videos, answers, items_per_rel)

    def _collect_data(self, videos, answers):
        true_data = []
        false_data = []
        for video in videos:
            for frame in video.frames:
                true_data_frame, false_data_frame = self._collect_objs(frame)
                true_data.extend(true_data_frame)
                false_data.extend(false_data_frame)

        # Sample equally from each set
        sample_num = self.items_per_rel // 2
        false_sample_num = self.items_per_rel - sample_num
        true_sampled = random.choices(true_data, k=sample_num)
        false_sampled = random.choices(false_data, k=false_sample_num)
        data = true_sampled + false_sampled
        random.shuffle(data)

        sampled_dict = {"close": data}
        return sampled_dict

    def _collect_objs(self, frame):
        true_data = []
        false_data = []
        for obj1 in frame.objs:
            for obj2 in frame.objs:
                obj_enc = self._obj_encoding(obj1)
                obj_enc.extend(self._obj_encoding(obj2))
                obj_enc = torch.tensor(obj_enc)
                if obj1.pos != obj2.pos and self._close_to(obj1, obj2):
                    true_data.append((obj_enc, torch.ones(1)))
                elif obj1.pos != obj2.pos:
                    false_data.append((obj_enc, torch.zeros(1)))

        return true_data, false_data

    def _close_to(self, obj1, obj2):
        """
        Returns whether the obj1 is close to the obj2
        A border is created around the obj1, obj2 is close if it is within the border

        :param obj1: Object 1
        :param obj2: Object 2
        :return: bool
        """

        obj1_x1, obj1_y1, obj1_x2, obj1_y2 = obj1.pos
        obj1_x1 -= self.close_to_def
        obj1_x2 += self.close_to_def
        obj1_y1 -= self.close_to_def
        obj1_y2 += self.close_to_def

        x1, y1, x2, y2 = obj2.pos
        obj_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        x_overlap = x2 >= obj1_x2 and x1 <= obj1_x1
        y_overlap = y2 >= obj1_y2 and y1 <= obj1_y1

        for x, y in obj_corners:
            match_x = obj1_x1 <= x <= obj1_x2 or x_overlap
            match_y = obj1_y1 <= y <= obj1_y2 or y_overlap
            if match_x and match_y:
                return True

        return False

    @staticmethod
    def from_video_dataset(spec, dataset, items_per_rel=DEFAULT_ITEMS_PER_REL):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        relation_data = HardcodedRelationDataset(spec, videos, answers, items_per_rel=items_per_rel)
        return relation_data
