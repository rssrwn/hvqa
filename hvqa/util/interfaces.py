# *** File for interfaces common to multiple objects ***


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
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


class Component(Trainable):
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

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the Component individually

        :param train_data: VideoDataset obj for training
        :param eval_data: VideoDataset obj for evaluation
        :param verbose: Print additional info during training
        """

        raise NotImplementedError()


class Model(Trainable):
    def run(self, frames, questions, q_types):
        """
        Generate answers to given questions

        :param frames: List of PIL Images
        :param questions: Questions about video: [str]
        :param q_types: Type of each question: [int]
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

        :param train_data: List of training Video objects
        :param eval_data: List of eval Video objects
        :param verbose: Print additional info during training
        """

        raise NotImplementedError()

    @staticmethod
    def new(spec, detector, **kwargs):
        """
        Create a new instance of the Model

        :param spec: Environment specification (dict)
        :param detector: Object detector (instance of Detector interface)
        :param kwargs: Other model specific params
        :return: Instance of Model
        """

        raise NotImplementedError()

    @staticmethod
    def load(path, **kwargs):
        """
        Load the Model object

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

        :param data: VideoDataset obj of evaluation data
        :param verbose: Boolean
        """

        raise NotImplementedError()
