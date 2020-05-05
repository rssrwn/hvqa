import random

from torch.utils.data import Dataset


class VideoPropDataset(Dataset):
    def __init__(self, spec, videos, transform=None, num_obj_per_cls=None):
        """
        Create a dataset for property extraction

        :param spec: EnvSpec object
        :param videos: List of Video objects
        :param transform: Torchvision Transform to apply to PIL image of object
        :param num_obj_per_cls: Number of objects from each class (sampled randomly)
        """

        self.spec = spec
        self.transform = transform

        # This needs to be run after setting the others
        self.obj_data = self._collect_data(videos, num_obj_per_cls)

        super(VideoPropDataset, self).__init__()

    def __len__(self):
        return len(self.obj_data)

    def __getitem__(self, item):
        img, cls, target = self.obj_data[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, cls, target

    def _collect_data(self, videos, num_obj):
        data = []
        cls_data = {cls: [] for cls in self.spec.obj_types()}
        for video in videos:
            for frame in video.frames:
                if num_obj is None:
                    [data.append((obj.img, obj.cls, obj.prop_vals)) for obj in frame.objs]
                else:
                    [cls_data[obj.cls].append((obj.img, obj.cls, obj.prop_vals)) for obj in frame.objs]

        # Sample data evenly from each class
        if num_obj is not None:
            for cls, items in cls_data.items():
                sampled = random.choices(items, k=num_obj)
                data.extend(sampled)

        return data

    @classmethod
    def from_video_dataset(cls, spec, dataset, transform=None, num_obj=None):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        prop_dataset = cls(spec, videos, transform=transform, num_obj_per_cls=num_obj)
        return prop_dataset


class QAPropDataset(Dataset):
    def __init__(self, spec, videos, answers, transform=None):
        self.spec = spec
        self.videos = videos
        self.transform = transform

        assert len(videos) == len(answers), "There must be an equal number of videos and answer lists"
        for idx, video in enumerate(videos):
            assert_str = "Each video should have the same number of questions as answers"
            assert len(video.questions) == len(answers[idx]), assert_str

        self.imgs, self.qa_classes, self.props = self._collect_data(videos, answers)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        imgs = self.imgs[item]
        cls = self.qa_classes[item]
        props = self.props[item]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        return imgs, cls, props

    @staticmethod
    def from_video_dataset(spec, dataset, transform=None):
        data = [dataset[idx] for idx in range(len(dataset))]
        videos, answers = tuple(zip(*data))
        prop_dataset = QAPropDataset(spec, videos, answers, transform=transform)
        return prop_dataset

    def _collect_data(self, videos, answers):
        imgs = []
        cls = []
        props = []
        for idx, video in enumerate(videos):
            video_ans = answers[idx]

            # TODO Move this
            prop_q_type = 0
            q_idxs = [idx for idx, q_type in enumerate(video.q_types) if q_type == prop_q_type]

            for q_idx in q_idxs:
                question = video.questions[q_idx]
                ans = video_ans[q_idx]
                prop, val, q_cls, frame_idx = self._parse_question(question)

                frame = video.frames[frame_idx]
                frame_imgs = [obj.img for obj in frame.objs if obj.cls == q_cls]

                # TODO Fix dataset
                q_props = {prop: self.spec.from_internal(prop, int(ans)) if prop == "rotation" else ans}
                if val is not None:
                    val_prop = self.spec.find_prop(val)
                    q_props[val_prop] = val

                imgs.append(frame_imgs)
                cls.append(q_cls)
                props.append(q_props)

        return imgs, cls, props

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
