from hvqa.models.abs_model import _AbsModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem


# TODO
# class IndividuallyTrainedModel(_AbsModel):
#     def __init__(self, events_path, qa_path, err_corr=True, detector_path=None, properties_path=None):
#         self.err_corr = err_corr
#
#         self.events_path = events_path
#         self.qa_path = qa_path
#
#         if detector_path is not None:
#             detector = NeuralDetector.load(detector_path)
#         else:
#             detector = NeuralDetector.new()
#
#         if properties_path is not None:
#             properties = NeuralPropExtractor.load(properties_path)
#         else:
#             properties = NeuralPropExtractor.new()
#
#         tracker = ObjTracker(err_corr)
#         # relations = HardcodedRelationClassifier()
#         # events = ASPEventDetector(events_path)
#         qa = HardcodedASPQASystem(qa_path)
#
#         # This will store each component
#         super(IndividuallyTrainedModel, self).__init__(
#             detector,
#             properties,
#             tracker,
#             relations,
#             events,
#             qa
#         )
#
#     @staticmethod
#     def load(path):
#         pass
#
#     def save(self, path):
#         pass
