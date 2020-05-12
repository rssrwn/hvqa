from hvqa.events.abs_asp import _AbsEventDetector
from hvqa.util.interfaces import Trainable


class ILASPEventDetector(_AbsEventDetector, Trainable):
    def __init__(self, spec):
        super(ILASPEventDetector, self).__init__(spec)

    @staticmethod
    def new(spec, **kwargs):
        events = ILASPEventDetector(spec)
        return events

    @staticmethod
    def load(spec, path):
        pass

    def save(self, path):
        pass

    def _detect_events(self, frames):
        pass

    def train(self, train_data, eval_data, verbose=True):
        pass
