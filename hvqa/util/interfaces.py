# *** File for interfaces common to multiple objects ***


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class Component:
    def run(self, data):
        """
        Run the component with some input <data>
        The input and output can take any type (it is up to the component to check this)

        :param data: Input to the component
        :return: Output of the component
        """

        raise NotImplementedError()

    def train(self, data):
        """
        Train the component with some training data <data>
        The input can take any type
        Nothing is returned

        :param data: Training data
        """

        raise NotImplementedError()

    def load(self, path):
        """
        Load the component from a file

        :param path: Path to load from (str or Path obj)
        """

        raise NotImplementedError()

    def save(self, path):
        """
        Save the component to a file

        :param path: Path to save to (str or Path obj)
        """

        raise NotImplementedError()


class Model:
    def run(self, frames, questions, q_types):
        """
        Generate answers to given questions

        :param frames: List of PIL Images
        :param questions: Questions about video: [str]
        :param q_types: Type of each question: [int]
        :return: Answers: [str]
        """

        raise NotImplementedError()

    def train(self, data, verbose=True):
        """
        Train the model

        :param data: VideoDataset obj of training data
        :param verbose: Boolean
        """

        raise NotImplementedError()

    def eval(self, data, verbose=True):
        """
        Evaluate the performance of the model

        :param data: VideoDataset obj of evaluation data
        :param verbose: Boolean
        """

        raise NotImplementedError()
