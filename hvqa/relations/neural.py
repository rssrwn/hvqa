from hvqa.util.interfaces import Component, Trainable


class NeuralRelationClassifier(Component, Trainable):
    def __init__(self):
        super(NeuralRelationClassifier, self).__init__()

        pass

    def run_(self, video):
        pass

    @staticmethod
    def load(spec, path):
        pass

    @staticmethod
    def new(spec, **kwargs):
        pass

    def train(self, train_data, eval_data, verbose=True):
        pass
