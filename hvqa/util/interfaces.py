# *** File for interfaces common to multiple objects ***

from torch.utils.data import Dataset


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class QADataset(Dataset):
    def __len__(self):
        """
        Get the length of the dataset

        :return: Number of items in the dataset (int)
        """

        raise NotImplementedError()

    def __getitem__(self, item):
        """
        Get an item from the dataset

        :param item: Idx of item in the dataset
        :return: Pair of Video and answers (Video, [str])
        """

        raise NotImplementedError()

    def is_hardcoded(self):
        """
        Return whether the dataset contains hardcoded data or requires the Model to fill in data itself
        The hardcoded data should mostly be object properties

        :return: Bool
        """

        raise NotImplementedError()

    def detector_timing(self):
        """
        Return the time taken by the detector (for debug and informational purposes)

        :return: Seconds
        """

        raise NotImplementedError()


class Trainable:
    def train(self, train_data, eval_data, verbose=True):
        """
        Train the component with some training data <data>
        The input can take any type
        Nothing is returned

        :param train_data: Training data
        :param eval_data: Eval data
        :param verbose: Print additional info during training
        """

        raise NotImplementedError()

    @staticmethod
    def load(spec, path):
        """
        Create a object by loading from a path

        :param spec: EnvSpec object
        :param path: Path to load from (str)
        :return: Instance of Trainable interface
        """

        raise NotImplementedError()


class Detector(Trainable):
    def detect_objs(self, frames):
        """
        Detect objects in the list of frames returning an Video obj

        :param frames: List of PIL images
        :return: Video object
        """

        raise NotImplementedError()

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError()

    @staticmethod
    def load(spec, path):
        raise NotImplementedError()


class Component:
    def run_(self, video):
        """
        Run the component for a given video
        Note: The function should modify the video obj in-place with the new information from the component

        :param video: Video obj
        """

        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        """
        Create a new instance of a Component

        :param spec: Environment specification
        :param kwargs: Other component specific params
        :return: Instance of the Component
        """

        raise NotImplementedError()


class Model(Trainable):
    def run(self, video):
        """
        Generate answers to given questions

        :param video: Video object
        :return: Answers: [str]
        """

        raise NotImplementedError()

    def process(self, frames):
        """
        Process the frames of a video and create a Video obj with the info extracted

        :param frames: List of PIL images
        :return: Video obj
        """

        raise NotImplementedError()

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the Model

        :param train_data: QADataset object
        :param eval_data: QADataset object
        :param verbose: Print additional info during training
        """

        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        """
        Create a new instance of the Model

        :param spec: Environment specification (dict)
        :param kwargs: Other model specific params
        :return: Instance of Model
        """

        raise NotImplementedError()

    @staticmethod
    def load(spec, path, **kwargs):
        """
        Load the Model object

        :param spec: EnvSpec object
        :param path: Path to where the object information was saved
        :return: Model obj
        """

        raise NotImplementedError()

    def save(self, path):
        """
        Save the Model information

        :param path: Path to where the model should be saved
        """

        raise NotImplementedError()

    def eval(self, data, verbose=True):
        """
        Evaluate the performance of the model

        :param data: QADataset obj of evaluation data
        :param verbose: Boolean
        """

        raise NotImplementedError()
