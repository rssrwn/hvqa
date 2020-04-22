from hvqa.models.abs_model import _AbsModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.hardcoded_relations import HardcodedRelationClassifier
from hvqa.events.asp_event_detector import ASPEventDetector
from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem


class HardcodedModel(_AbsModel):
    def __init__(self, detector_path, prop_classifier_path, event_asp_dir, qa_system_asp_dir):
        self.tracker_err_correction = True

        # This will set paths and setup components by calling the setup functions in this class
        super(HardcodedModel, self).__init__(
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
        tracker = ObjTracker(err_corr=self.tracker_err_correction)
        return tracker

    def _setup_relation_classifier(self):
        relations = HardcodedRelationClassifier()
        return relations

    def _setup_event_detector(self):
        event_detector = ASPEventDetector(self.events_path)
        return event_detector

    def _setup_qa_system(self):
        qa = HardcodedASPQASystem(self.qa_path)
        return qa
