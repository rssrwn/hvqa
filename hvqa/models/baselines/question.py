from hvqa.util.interfaces import Model


class RandomAns(Model):
    def __init__(self):
        pass

    def run(self, video):
        pass

    def process(self, frames):
        pass

    def train(self, train_data, eval_data, verbose=True):
        pass

    @staticmethod
    def new(spec, **kwargs):
        pass

    @staticmethod
    def load(spec, path, **kwargs):
        pass

    def save(self, path):
        pass

    def eval(self, data, verbose=True):
        pass

    def eval_components(self, eval_data):
        pass
