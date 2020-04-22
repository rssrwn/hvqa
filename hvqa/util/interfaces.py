# *** File for interfaces common to multiple objects ***


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class Trainable:
    def train(self, data):
        """
        Train the component with some training data <data>
        The input can take any type
        Nothing is returned

        :param data: Training data
        """

        raise NotImplementedError()


class Component(Trainable):
    def run_(self, video):
        """
        Run the component for a given video
        Note: The function should modify the video obj in-place with the new information from the component

        :param video: Video obj
        """

        raise NotImplementedError()

    def train(self, data):
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

    def train(self, data, verbose=True):
        raise NotImplementedError()

    @staticmethod
    def load(path):
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
