# *** File for interfaces common to multiple objects ***


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()


class Component:
    def run(self, data):
        raise NotImplementedError()

    def train(self, data):
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
