from hvqa.models.abs_model import _AbsModel


class HardcodedModel(_AbsModel):
    def __init__(self):
        super(HardcodedModel, self).__init__()

    def _setup_detector(self):
        pass

    def _setup_prop_classifier(self):
        pass

    def _setup_tracker(self):
        pass

    def _setup_relation_classifier(self):
        pass

    def _setup_event_detector(self):
        pass

    def _setup_qa_system(self):
        pass
