import json
from pathlib import Path

from hvqa.models.abs_model import _AbsVQAModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.hardcoded_relations import HardcodedRelationClassifier
from hvqa.events.asp_event_detector import ASPEventDetector
from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem


ERR_CORR_DEFAULT = True
AL_MODEL_DEFAULT = True


class HardcodedVQAModel(_AbsVQAModel):
    def __init__(self, detector, properties, tracker, relations, events, qa):
        self.err_corr = tracker.err_corr
        self.al_model = events.al_model

        super(HardcodedVQAModel, self).__init__(detector, properties, tracker, relations, events, qa)

    def train(self, data, verbose=True):
        # TODO
        pass

    @staticmethod
    def new(spec, detector, params=None):
        err_corr = ERR_CORR_DEFAULT
        al_model = AL_MODEL_DEFAULT
        if params is not None:
            err_corr_param = params.get("error_correction")
            al_model_param = params.get("al_model")
            err_corr = err_corr_param if err_corr_param is not None else err_corr
            al_model = al_model_param if al_model_param is not None else al_model

        properties = NeuralPropExtractor(spec)
        tracker = ObjTracker(spec, err_corr)
        relations = HardcodedRelationClassifier(spec)
        events = ASPEventDetector(spec)
        qa = HardcodedASPQASystem(spec)

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

        err_corr = meta_data["err_corr"]
        al_model = meta_data["al_model"]

        detector = NeuralDetector.load(detector_path)
        properties = NeuralPropExtractor.load(properties_path)
        tracker = ObjTracker(err_corr)
        relations = HardcodedRelationClassifier()
        events = ASPEventDetector(al_model)
        qa = HardcodedASPQASystem()

        model = HardcodedVQAModel(detector, properties, tracker, relations, events, qa)
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
            "err_corr": self.err_corr,
            "al_model": self.al_model,
        }

        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)

        print(f"Successfully saved VideoQA model to {path}")
