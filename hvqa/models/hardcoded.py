from hvqa.models.abs_model import _AbsModel
from hvqa.detection.models import DetectionBackbone, DetectionModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.models import PropertyExtractionModel
from hvqa.properties.prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.obj_relations import HardcodedRelationClassifier
from hvqa.events.event_detector import ASPEventDetector
from hvqa.qa.qa_system import HardcodedASPQASystem
from hvqa.util.func import load_model


class HardcodedModel(_AbsModel):
    def __init__(self, detector_path, prop_classifier_path, event_asp_dir, qa_system_asp_dir):
        super(HardcodedModel, self).__init__()

        self.detector_path = detector_path
        self.prop_classifier_path = prop_classifier_path
        self.event_asp_dir = event_asp_dir
        self.qa_system_asp_dir = qa_system_asp_dir

    def _setup_detector(self):
        backbone = DetectionBackbone()
        detector_model = load_model(DetectionModel, self.detector_path, backbone)
        detector_model.eval()
        detector = NeuralDetector(detector_model)
        return detector

    def _setup_prop_classifier(self):
        prop_model = load_model(PropertyExtractionModel, self.prop_classifier_path)
        prop_model.eval()
        prop_extractor = NeuralPropExtractor(prop_model)
        return prop_extractor

    def _setup_tracker(self):
        tracker = ObjTracker()
        return tracker

    def _setup_relation_classifier(self):
        relations = HardcodedRelationClassifier()
        return relations

    def _setup_event_detector(self):
        event_detector = ASPEventDetector(self.event_asp_dir)
        return event_detector

    def _setup_qa_system(self):
        qa = HardcodedASPQASystem(self.qa_system_asp_dir)
        return qa
