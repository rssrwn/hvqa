from hvqa.models.abs_model import _AbsModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor


class IndividuallyTrainedModel(_AbsModel):
    def __init__(self, detector_path, prop_classifier_path, event_asp_dir, qa_system_asp_dir):
        self.tracker_err_correction = True

        # This will set paths and setup components by calling the setup functions in this class
        super(IndividuallyTrainedModel, self).__init__(
            detector_path,
            prop_classifier_path,
            None,
            None,
            event_asp_dir,
            qa_system_asp_dir
        )

    def _setup_obj_detector(self):
        detector = NeuralDetector.load(self.detector_path)
        return detector

    def _setup_prop_classifier(self):
        prop_extractor = NeuralPropExtractor.load(self.properties_path)
        return prop_extractor

    def _setup_tracker(self):
        pass

    def _setup_relation_classifier(self):
        pass

    def _setup_event_detector(self):
        pass

    def _setup_qa_system(self):
        pass
