import json
from pathlib import Path

from hvqa.util.environment import EnvSpec
from hvqa.models.abs_model import _AbsVQAModel
from hvqa.properties.neural_prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.hardcoded_relations import HardcodedRelationClassifier
from hvqa.events.asp_event_detector import ASPEventDetector
from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem


ERR_CORR_DEFAULT = True
AL_MODEL_DEFAULT = True


class HardcodedVQAModel(_AbsVQAModel):
    def __init__(self, spec, properties, tracker, relations, events, qa):
        self.spec = spec
        self.err_corr = tracker.err_corr
        self.al_model = events.al_model

        super(HardcodedVQAModel, self).__init__(properties, tracker, relations, events, qa)

    def train(self, train_data, eval_data, verbose=True):
        print("\nTraining hardcoded model...")
        self.prop_classifier.train(train_data, eval_data, verbose=verbose)
        print("Completed hardcoded model training.")

    def eval_components(self, eval_data):
        self.prop_classifier.eval(eval_data)

    @staticmethod
    def new(spec, **kwargs):
        err_corr = ERR_CORR_DEFAULT
        al_model = AL_MODEL_DEFAULT
        if kwargs is not None:
            err_corr_param = kwargs.get("err_corr")
            al_model_param = kwargs.get("al_model")
            err_corr = err_corr_param if err_corr_param is not None else err_corr
            al_model = al_model_param if al_model_param is not None else al_model

        properties = NeuralPropExtractor.new(spec)
        tracker = ObjTracker.new(spec, err_corr=err_corr)
        relations = HardcodedRelationClassifier.new(spec)
        events = ASPEventDetector.new(spec, al_model=al_model)
        qa = HardcodedASPQASystem.new(spec)

        model = HardcodedVQAModel(spec, properties, tracker, relations, events, qa)
        return model

    @staticmethod
    def load(path, **kwargs):
        """
        Loads the model using metadata from the json object saved at <path>
        The path should be a json file created by model.save(<path>)

        :param path: Path of json file to load model from (str)
        :return: _AbsModel
        """

        save_path = Path(path)
        properties_path = str(save_path / "properties.pt")
        meta_data_path = save_path / "meta_data.json"

        with meta_data_path.open() as f:
            meta_data = json.load(f)

        err_corr = meta_data["err_corr"]
        err_corr_param = kwargs.get("err_corr")
        err_corr = err_corr_param if err_corr_param is not None else err_corr

        al_model = meta_data["al_model"]
        al_model_param = kwargs.get("al_model")
        al_model = al_model_param if al_model_param is not None else al_model

        spec_dict = meta_data["spec"]
        spec = EnvSpec.from_dict(spec_dict)
        properties = NeuralPropExtractor.load(spec, properties_path)
        tracker = ObjTracker(err_corr)
        relations = HardcodedRelationClassifier()
        events = ASPEventDetector(al_model)
        qa = HardcodedASPQASystem()

        model = HardcodedVQAModel(spec, properties, tracker, relations, events, qa)
        return model

    def save(self, path):
        """
         This will create (or overwrite) a directory at <path>
         The directory contains saved files from the individual components and a Model meta-data file

         :param path: Path to save model to (str)
         """

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        properties_path = save_path / "properties.pt"
        meta_data_path = save_path / "meta_data.json"

        self.prop_classifier.save(str(properties_path))

        meta_data = {
            "err_corr": self.err_corr,
            "al_model": self.al_model,
            "spec": self.spec.to_dict()
        }

        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)

        print(f"Successfully saved VideoQA model to {path}")
