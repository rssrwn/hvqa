from torch.utils.data import Dataset


class PropDataset(Dataset):
    def __init__(self, spec, videos, hardcoded, transform=None):
        """
        Create a dataset for property extraction

        :param spec: EnvSpec object
        :param videos: List of Video objects
        :param hardcoded: Is object data in the videos hardcoded (bool)
        :param transform: Torchvision Transform to apply to PIL image of object
        """

        self.spec = spec
        self.hardcoded = hardcoded
        self.transform = transform
        self.obj_data = self._collect_data(videos)

        super(PropDataset, self).__init__()

    def __len__(self):
        return len(self.obj_data)

    def __getitem__(self, item):
        img, target = self.obj_data[item]
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    @staticmethod
    def from_qa_dataset(spec, dataset, transform=None):
        """
        Construct the properties dataset using the data in a given QADataset

        :param spec: EnvSpec object
        :param dataset: QADataset object
        :param transform: Torchvision Transform to apply to PIL image of object
        :return: PropDataset obj
        """

        hardcoded = dataset.is_hardcoded()
        videos = [dataset[idx] for idx in range(len(dataset))]
        prop_dataset = PropDataset(spec, videos, hardcoded, transform)
        return prop_dataset

    def _collect_data(self, videos):
        """
        Collect data for the dataset

        :param videos: List of Video objs
        :return: Data: tuple of (PIL Image, dict of {prop: val})
        """

        data = []
        for video in videos:
            for frame in video.frames:
                if self.hardcoded:
                    [data.append((obj.img, obj.prop_vals)) for obj in frame.objs]
                else:
                    raise NotImplementedError()

        return data

    # TODO create separate question parsing class
    def _parse_question(self, question):
        splits = question.split(" ")
        prop = splits[1]
        frame_idx = int(splits[-1][:-1])

        # Assume only one property value given in question
        cls = splits[4]
        prop_val = None
        if cls not in self.spec.obj_types():
            prop_val = cls
            cls = splits[5]

        return prop, prop_val, cls, frame_idx
