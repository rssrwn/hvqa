import json
from pathlib import Path

from hvqa.models.abs_model import _AbsVQAModel
from hvqa.properties.neural import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.relations.neural import NeuralRelationClassifier
from hvqa.events.trainable import ILPEventDetector
from hvqa.qa.hardcoded_asp import HardcodedASPQASystem


class IndTrainedModel(_AbsVQAModel):
    def __init__(self, spec, properties, tracker, relations, events, qa):
        self.spec = spec
        self.err_corr = tracker.err_corr
        # self.al_model = events.al_model

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
        # [self.prop_classifier.run_(video) for video in videos]  # TODO uncomment

        print("Adding tracking ids to objects...")
        [self.tracker.run_(video) for video in videos]

        # Train relation component and add relations to each frame
        # self.relation_classifier.train((videos, answers), eval_data, verbose=verbose)  # TODO uncomment

        print("Labelling relations between objects...")
        # [self.relation_classifier.run_(video) for video in videos]  # TODO uncomment

        print("Training event detector...")
        self.event_detector.train((videos, answers), eval_data, verbose=verbose)

        print("Completed individually-trained model training.")

    def eval_components(self, eval_data):
        print("\nEvaluating components of IndTrainedModel...")
        self.prop_classifier.eval(eval_data)
        self.relation_classifier.eval(eval_data)
        self.event_detector.eval(eval_data, self.tracker)
        print("Completed component evaluation.")

    @staticmethod
    def new(spec, **kwargs):
        properties = NeuralPropExtractor.new(spec)
        tracker = ObjTracker.new(spec, err_corr=False)
        relations = NeuralRelationClassifier.new(spec)
        events = ILPEventDetector.new(spec)
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
        events_path = str(save_path / "event_rules.json")
        meta_data_path = save_path / "meta_data.json"

        with meta_data_path.open() as f:
            meta_data = json.load(f)

        properties = NeuralPropExtractor.load(spec, properties_path)
        tracker = ObjTracker.new(spec, err_corr=False)
        relations = NeuralRelationClassifier.load(spec, relations_path)

        # events = ILPEventDetector.load(spec, events_path)
        events = ILPEventDetector.new(spec)  # TODO update to load

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
        events_path = save_path / "event_rules.json"
        meta_data_path = save_path / "meta_data.json"

        self.prop_classifier.save(str(properties_path))
        self.relation_classifier.save(str(relations_path))
        self.event_detector.save(str(events_path))

        meta_data = {
            "spec": self.spec.to_dict()
        }

        with open(meta_data_path, "w") as f:
            json.dump(meta_data, f)

        print(f"Successfully saved VideoQA model to {path}")
