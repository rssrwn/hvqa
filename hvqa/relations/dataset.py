import random

import torch
from torch.utils.data import Dataset


DEFAULT_ITEMS_PER_REL = 10000


class RelationDataset(Dataset):
    def __init__(self, spec, videos, answers, items_per_rel=DEFAULT_ITEMS_PER_REL):
        self.spec = spec
        self.items_per_rel = items_per_rel

        self.rel_data_map = self._collect_data(videos, answers)

    def __len__(self):
        return self.items_per_rel

    def __getitem__(self, item):
        rel_data = {rel: self.rel_data_map[rel][item] for rel in self.spec.relations}
        return rel_data

    def _collect_data(self, videos, answers):
        rel_data_map = {rel: [] for rel in self.spec.relations}
        for video_idx, video in enumerate(videos):
            q_idxs = [idx for idx, q_type in enumerate(video.q_types) if q_type == self.spec.qa.relation_q]
            for q_idx in q_idxs:
                question = video.questions[q_idx]
                answer = answers[video_idx][q_idx]

                rel, obj1_cls, obj1_val, obj2_cls, obj2_val, frame_idx = self.spec.qa.parse_relation_question(question)
                obj1_prop = self.spec.find_prop(obj1_val)
                obj2_prop = self.spec.find_prop(obj2_val)
                yes_no = self.spec.qa.parse_q_1(answer)
                yes_no = 1.0 if yes_no == "yes" else 0.0

                obj1 = None
                obj2 = None
                for obj in video.frames[frame_idx].objs:
                    if obj.prop_vals[obj1_prop] == obj1_val:
                        obj1 = obj
                    if obj.prop_vals[obj2_prop] == obj2_val:
                        obj2 = obj

                assert obj1 is not None, f"Cannot find obj with {obj1_prop} {obj1_val} in frame {frame_idx}"
                assert obj2 is not None, f"Cannot find obj with {obj2_prop} {obj2_val} in frame {frame_idx}"

                obj_enc = self._obj_encoding(obj1)
                obj2_enc = self._obj_encoding(obj2)
                obj_enc.extend(obj2_enc)

                objs = torch.tensor(obj_enc)
                classification = torch.tensor(yes_no)
                rel_data_map[rel].append((objs, classification))

        sampled_rel_data_map = {}
        for rel, data in rel_data_map.items():
            sampled = random.choices(data, k=self.items_per_rel)
            sampled_rel_data_map[rel] = sampled

        return sampled_rel_data_map

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

    @staticmethod
    def from_video_dataset(spec, dataset, items_per_rel=DEFAULT_ITEMS_PER_REL):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        relation_data = RelationDataset(spec, videos, answers, items_per_rel=items_per_rel)
        return relation_data
