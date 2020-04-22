import json
from pathlib import Path

from hvqa.models.abs_model import _AbsModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.hardcoded_relations import HardcodedRelationClassifier
from hvqa.events.asp_event_detector import ASPEventDetector
from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem


class HardcodedModel(_AbsModel):
    def __init__(self, events_path, qa_path, err_corr=True, detector_path=None, properties_path=None):
        self.err_corr = err_corr

        self.events_path = events_path
        self.qa_path = qa_path

        if detector_path is not None:
            detector = NeuralDetector.load(detector_path)
        else:
            detector = NeuralDetector.new()

        if properties_path is not None:
            properties = NeuralPropExtractor.load(properties_path)
        else:
            properties = NeuralPropExtractor.new()

        tracker = ObjTracker(err_corr)
        relations = HardcodedRelationClassifier()
        events = ASPEventDetector(events_path)
        qa = HardcodedASPQASystem(qa_path)

        # This will store each component
        super(HardcodedModel, self).__init__(
            detector,
            properties,
            tracker,
            relations,
            events,
            qa
        )

    @staticmethod
    def load(path):
        """
        Loads the model using metadata from the json object saved at <path>
        The path should be a json file created by model.save(<path>)

        :param path: Path of json file to load model from (str)
        :return: _AbsModel
        """

        save_path = Path(path)

        detector_path = str(save_path / "detector.pt")
        properties_path = str(save_path / "properties.pt")
        meta_data_path = save_path / "meta_data.json"

        with meta_data_path.open() as f:
            meta_data = json.load(f)

        events_path = meta_data["events"]
        qa_path = meta_data["qa"]
        err_corr = meta_data["error_correction"]

        model = HardcodedModel(events_path, qa_path, err_corr, detector_path, properties_path)
        return model

    def save(self, path):
        """
         This will create (or overwrite) a json file at <path>
         Note: The individual components will be saved at the paths specified when the model was created
               Not the paths that the components may have been loaded from

         :param path: Path to save json file to (str)
         """

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        detector_path = save_path / "detector.pt"
        properties_path = save_path / "properties.pt"
        meta_data_path = save_path / "meta_data.json"

        self.obj_detector.save(str(detector_path))
        self.prop_classifier.save(str(properties_path))

        meta_data = {
            "events": self.events_path,
            "qa": self.qa_path,
            "error_correction": self.err_corr
        }

        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)

        print(f"Successfully saved VideoQA model to {path}")
