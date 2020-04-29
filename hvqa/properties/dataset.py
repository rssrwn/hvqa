import random
from torch.utils.data import Dataset

from hvqa.util.func import append_in_map


class VideoPropDataset(Dataset):
    def __init__(self, spec, videos, transform=None):
        """
        Create a dataset for property extraction

        :param spec: EnvSpec object
        :param videos: List of Video objects
        :param transform: Torchvision Transform to apply to PIL image of object
        """

        self.spec = spec
        self.transform = transform

        # This needs to be run after setting the others
        self.obj_data = self._collect_data(videos)

        super(VideoPropDataset, self).__init__()

    def __len__(self):
        return len(self.obj_data)

    def __getitem__(self, item):
        img, target = self.obj_data[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    @staticmethod
    def _collect_data(videos):
        data = []
        for video in videos:
            for frame in video.frames:
                [data.append((obj.img, obj.prop_vals)) for obj in frame.objs]

        return data

    @classmethod
    def from_video_dataset(cls, spec, dataset, transform=None):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        prop_dataset = cls(spec, videos, transform=transform)
        return prop_dataset


class QAPropDataset(Dataset):
    def __init__(self, spec, videos, answers=None, transform=None):
        """
        Create a dataset for property extraction

        :param spec: EnvSpec object
        :param videos: List of Video objects
        :param answers: List of list of answers ([[str]])
        :param transform: Torchvision Transform to apply to PIL image of object
        """

        self.spec = spec
        self.transform = transform
        self.answers = answers

        # This needs to be run after setting the others
        self.obj_data, self.num_items = self._collect_data(videos)

        super(QAPropDataset, self).__init__()

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        targets = {prop: items[item] for prop, items in self.obj_data.items()}
        targets = {prop: (self.transform(img), target) for prop, (img, target) in targets.items()}
        return targets

    def _collect_data(self, videos):
        objs = {}
        for video_idx, video in enumerate(videos):
            questions = video.questions
            q_types = video.q_types
            answers = self.answers[video_idx]
            frames = video.frames

            # TODO put this in spec (or a new question spec obj)
            prop_q_type = 0

            q_idxs = [idx for idx, q_type in enumerate(q_types) if q_type == prop_q_type]
            for q_idx in q_idxs:
                question = questions[q_idx]
                answer = answers[q_idx]
                obj_tuple = self._collect_obj(frames, question, answer)
                if obj_tuple is not None:
                    obj_cls, img, targets = obj_tuple
                    append_in_map(objs, obj_cls, (img, targets))

        print(f"{'Class':<15}{'Num Objs'}")
        for cls, items in objs.items():
            print(f"{cls:<15}{len(items)}")

        # Sample from each class equally
        sampled_data = []
        items_per_cls = max([len(items) for cls, items in objs.items()])
        for cls, items in objs.items():
            if len(items) == items_per_cls:
                sampled_data.extend(items)
            else:
                idxs = random.choices(range(len(items)), k=items_per_cls)
                sampled = [items[idx] for idx in idxs]
                sampled_data.extend(sampled)

        prop_map = {}
        for img, targets in sampled_data:
            for prop, val in targets.items():
                append_in_map(prop_map, prop, (img, {prop: val}))

        # Sample from each property equally
        sampled_prop_map = {}
        num_items = max([len(items) for prop, items in prop_map.items()])
        for prop, items in prop_map.items():
            if len(items) == num_items:
                sampled_prop_map[prop] = items
            else:
                sampled_prop_map[prop] = random.choices(items, k=num_items)

        return sampled_prop_map, num_items

    def _collect_obj(self, frames, question, answer):
        objs = []
        prop, val, cls, frame_idx = self._parse_question(question)
        frame = frames[frame_idx]
        for obj in frame.objs:
            obj_val = None
            if val is not None:
                prop_for_val = self.spec.find_prop(val)
                obj_val = obj.prop_vals.get(prop_for_val)

            # Match an obj (if val is None we can still match if data has not been loaded)
            if obj.cls == cls and val == obj_val:

                # TODO remove
                # if prop == "rotation":
                #     answer = self.spec.from_internal(prop, int(answer))

                target = {prop: answer}

                # Add the val given in the question as well (if we can)
                if val is not None:
                    target[prop_for_val] = val

                objs.append((cls, obj.img, target))

        if len(objs) == 0:
            return None

        obj = objs[0]
        if len(objs) > 1:
            print(f"WARNING: Found multiple objects for frame {frame_idx}, question {question}")

        return obj

    # TODO create separate question parsing class
    def _parse_question(self, question):
        splits = question.split(" ")
        prop = splits[1]
        frame_idx = int(splits[-1][:-1])

        # Assume only one property value given in question
        cls = splits[4]
        val = None
        if cls not in self.spec.obj_types():
            val = cls
            cls = splits[5]

        return prop, val, cls, frame_idx
