# *** File for interfaces common to multiple objects ***


class _AbsEvaluator:
    def __init__(self, test_loader):
        self.test_loader = test_loader

    def eval_model(self, model, threshold):
        raise NotImplementedError()
