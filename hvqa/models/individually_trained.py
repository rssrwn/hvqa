import json
from pathlib import Path

from hvqa.util.func import get_or_default
from hvqa.models.abs_model import _AbsVQAModel
from hvqa.properties.neural import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.neural import NeuralRelationClassifier
from hvqa.events.hardcoded_asp import ASPEventDetector
from hvqa.qa.hardcoded_asp import HardcodedASPQASystem


ERR_CORR_DEFAULT = True
AL_MODEL_DEFAULT = True


class IndTrainedModel(_AbsVQAModel):
    def __init__(self, spec, properties, tracker, relations, events, qa):
        self.spec = spec
        self.err_corr = tracker.err_corr
        self.al_model = events.al_model

        super(IndTrainedModel, self).__init__(properties, tracker, relations, events, qa)

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the Model

        :param train_data: QADataset object
        :param eval_data: QADataset object
        :param verbose: Print additional info during training
        """

        assert not train_data.is_hardcoded(), "Training data must not be hardcoded when training an IndTrainedModel"
        assert eval_data.is_hardcoded(), "Evaluation data must always be hardcoded when training an IndTrainedModel"

        print("\nTraining individually-trained model...")

        data = [train_data[idx] for idx in range(len(train_data))]
        videos, answers = tuple(zip(*data))

        # Train property component and label all objects with their properties
        # self.prop_classifier.train((videos, answers), eval_data, verbose=verbose, from_qa=True)  # TODO uncomment

        print("Labelling object properties...")
        [self.prop_classifier.run_(video) for video in videos]

        # Train relation component and add relations to each frame
        self.relation_classifier.train((videos, answers), eval_data, verbose=verbose)

        print("Labelling relations between objects...")
        # [self.relation_classifier.run_(video) for video in videos]

        print("Completed individually-trained model training.")

    def eval_components(self, eval_data):
        print("\nEvaluating components of IndTrainedModel...")
        self.prop_classifier.eval(eval_data)
        self.relation_classifier.eval(eval_data)
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
        relations = NeuralRelationClassifier.new(spec)
        events = ASPEventDetector.new(spec, al_model=al_model)
        qa = HardcodedASPQASystem.new(spec)

        model = IndTrainedModel(spec, properties, tracker, relations, events, qa)
        return model

    @staticmethod
    def load(spec, path, **kwargs):
        """
        Loads the model using metadata from the json object saved at <path>
        The path should be a json file created by model.save(<path>)

        :param spec: EnvSpec object
        :param path: Path of json file to load model from (str)
        :return: IndTrainedModel object
        """

        save_path = Path(path)
        properties_path = str(save_path / "properties.pt")
        relations_path = str(save_path / "relations.pt")
        meta_data_path = save_path / "meta_data.json"

        with meta_data_path.open() as f:
            meta_data = json.load(f)

        err_corr = get_or_default(kwargs, meta_data, "err_corr")
        al_model = get_or_default(kwargs, meta_data, "al_model")

        properties = NeuralPropExtractor.load(spec, properties_path)
        tracker = ObjTracker.new(spec, err_corr=err_corr)

        # relations = NeuralRelationClassifier.load(spec, relations_path)
        relations = NeuralRelationClassifier.new(spec)  # TODO remove

        events = ASPEventDetector.new(spec, al_model=al_model)
        qa = HardcodedASPQASystem.new(spec)

        model = IndTrainedModel(spec, properties, tracker, relations, events, qa)
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
        relations_path = save_path / "relations.pt"
        meta_data_path = save_path / "meta_data.json"

        self.prop_classifier.save(str(properties_path))
        self.relation_classifier.save(str(relations_path))

        meta_data = {
            "err_corr": self.err_corr,
            "al_model": self.al_model,
            "spec": self.spec.to_dict()
        }

        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)

        print(f"Successfully saved VideoQA model to {path}")
