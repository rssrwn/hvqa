import json
from pathlib import Path

from hvqa.util.func import get_or_default
from hvqa.models.abs_model import _AbsVQAModel
from hvqa.properties.neural import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.hardcoded import HardcodedRelationClassifier
from hvqa.events.hardcoded_asp import ASPEventDetector
from hvqa.qa.hardcoded_asp import HardcodedASPQASystem


ERR_CORR_DEFAULT = True
AL_MODEL_DEFAULT = True


class HardcodedVQAModel(_AbsVQAModel):
    def __init__(self, spec, properties, tracker, relations, events, qa):
        self.spec = spec
        self.err_corr = tracker.err_corr
        self.al_model = events.al_model

        super(HardcodedVQAModel, self).__init__(properties, tracker, relations, events, qa)

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the Model

        :param train_data: QADataset object
        :param eval_data: QADataset object
        :param verbose: Print additional info during training
        """

        assert train_data.is_hardcoded(), "Training data must be hardcoded when training HardcodedVQAModel"
        assert eval_data.is_hardcoded(), "Evaluation must be hardcoded when training HardcodedVQAModel"

        print("\nTraining hardcoded model...")

        self.prop_classifier.train(train_data, eval_data, verbose=verbose, from_qa=False)

        print("Completed hardcoded model training.")

    def eval_components(self, eval_data):
        print("\nEvaluating components of HardcodedModel...")
        # self.prop_classifier.eval(eval_data)
        self.event_detector.eval(eval_data, self.tracker)
        print("Completed component evaluation.")

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
    def load(spec, path, **kwargs):
        """
        Loads the model using metadata from the json object saved at <path>
        The path should be a json file created by model.save(<path>)

        :param spec: EnvSpec object
        :param path: Path of json file to load model from (str)
        :return: HardcodedVQAModel
        """

        save_path = Path(path)
        properties_path = str(save_path / "properties.pt")
        meta_data_path = save_path / "meta_data.json"

        with meta_data_path.open() as f:
            meta_data = json.load(f)

        err_corr = get_or_default(kwargs, meta_data, "err_corr")
        al_model = get_or_default(kwargs, meta_data, "al_model")

        properties = NeuralPropExtractor.load(spec, properties_path)
        tracker = ObjTracker.new(spec, err_corr=err_corr)
        relations = HardcodedRelationClassifier.new(spec)
        events = ASPEventDetector.new(spec, al_model=al_model)
        qa = HardcodedASPQASystem.new(spec)

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
