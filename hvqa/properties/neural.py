import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine
from more_itertools import grouper

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader

from hvqa.properties.dataset import VideoPropDataset, QAPropDataset
from hvqa.properties.models import PropertyExtractionModel, ObjectAutoEncoder
from hvqa.util.func import get_device, load_model, save_model, collate_func, append_in_map
from hvqa.util.interfaces import Component, Trainable
from hvqa.util.asp_runner import ASPRunner
from hvqa.util.dataset import VideoDataset


_transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
])

_ae_transform = T.Compose([
    T.Resize((16, 16)),
    T.ToTensor(),
])


# TODO Add evaluation of component with AE, before Prop network is trained

class NeuralPropExtractor(Component, Trainable):
    def __init__(self, spec, model, hardcoded=True):
        super(NeuralPropExtractor, self).__init__()

        self.device = get_device()

        self.spec = spec
        self.model = model.to(self.device)
        self.hardcoded = hardcoded

        self.loss_fn = nn.CrossEntropyLoss()
        self._loss_fn_ae = nn.L1Loss(reduction="none")

        self._temp_save = Path("saved-models/properties/temp")
        self._temp_save.mkdir(exist_ok=True, parents=True)

        self._temp_asp_file = "hvqa/properties/.temp_qa_training.lp"

    def run_(self, video):
        for frame in video.frames:
            objs = frame.objs
            obj_imgs = [obj.img for obj in objs]
            props = self._extract_props(obj_imgs)
            for idx, prop_vals in enumerate(props):
                obj = objs[idx]
                for prop, val in prop_vals.items():
                    obj.set_prop_val(prop, val)

    def _extract_props(self, obj_imgs):
        """
        Extracts properties of objects from images

        :param obj_imgs: List of PIL images of objects
        :return: List of dicts [{prop: val}]
        """

        device = get_device()
        obj_imgs = [_transform(img) for img in obj_imgs]
        obj_imgs_batch = torch.stack(obj_imgs)
        obj_imgs_batch = obj_imgs_batch.to(device)

        with torch.no_grad():
            model_out = self.model(obj_imgs_batch)

        batch_sizes = set([len(out) for _, out in model_out.items()])
        assert len(batch_sizes) == 1, "Number of model predictions must be the same for each property"

        objs = []
        for idx in range(list(batch_sizes)[0]):
            obj = {}
            for prop, pred in model_out.items():
                pred = torch.max(pred, dim=1)[1].cpu().numpy()
                val = self.spec.from_internal(prop, pred[idx])
                obj[prop] = val

            objs.append(obj)

        return objs

    @staticmethod
    def new(spec, **kwargs):
        model = PropertyExtractionModel(spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    @staticmethod
    def load(spec, path):
        model = load_model(PropertyExtractionModel, path, spec)
        model.eval()
        prop_extractor = NeuralPropExtractor(spec, model)
        return prop_extractor

    def save(self, path):
        save_model(self.model, path)

    def train(self, train_data, eval_data, verbose=True, from_qa=False):
        """
        Train the property classification component

        :param train_data: Training data ((Videos, answers))
        :param eval_data: Evaluation data (QADataset)
        :param verbose: Verbose printing (bool)
        :param from_qa: Train using data from QA pairs only (bool)
        """

        if from_qa:
            self._train_from_qa(train_data, eval_data, verbose)
        else:
            self._train_from_hardcoded(train_data, eval_data, verbose)

    def eval(self, eval_data, threshold=0.5, batch_size=256):
        """
        Evaluate the trained property component individually

        :param eval_data: Evaluation data (VideoDataset)
        :param threshold: Threshold for accepting a property classification
        :param batch_size: Maximum number of objects to pass through network at once
        """

        assert eval_data.is_hardcoded(), "VideoQADataset must be hardcoded when evaluating the NeuralPropExtractor"

        print("Evaluating neural property extraction component...")

        eval_dataset = VideoPropDataset.from_video_dataset(self.spec, eval_data, transform=_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)

        self.model.eval()

        # Setup metrics
        num_predictions = 0
        props = self.spec.prop_names()
        losses = {prop: [] for prop in props}
        tps = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        fps = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        tns = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        fns = {prop: {val: 0 for val in self.spec.prop_values(prop)} for prop in props}
        correct = {prop: 0 for prop in props}

        for i, (imgs, _, objs) in enumerate(eval_loader):
            images, targets = self._prepare_data(imgs, objs)

            with torch.no_grad():
                preds = self.model(images)

            # TODO apply softmax before threshold
            output = {prop: out.to("cpu") for prop, out in preds.items()}
            targets = {prop: target.to("cpu") for prop, target in targets.items()}
            results = {prop: self._eval_prop(pred, targets[prop], threshold) for prop, pred in output.items()}

            for prop, (loss_, tps_, fps_, tns_, fns_, num_correct) in results.items():
                for idx, target in enumerate(self.spec.prop_values(prop)):
                    losses[prop].extend(loss_)
                    tps[prop][target] += tps_[idx]
                    fps[prop][target] += fps_[idx]
                    tns[prop][target] += tns_[idx]
                    fns[prop][target] += fns_[idx]

                correct[prop] += num_correct
            num_predictions += len(imgs)

        results = (tps, fps, tns, fns)
        self._print_results(results, correct, losses, num_predictions)

    def _train_from_hardcoded(self, train_data, eval_data, verbose, lr=0.001, batch_size=256, epochs=2):
        videos, answers = train_data
        train_dataset = VideoPropDataset(self.spec, videos, transform=_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
        optimiser = optim.Adam(self.model.parameters(), lr=lr)

        print(f"Training property extraction model using object data from VideoDataset with device {self.device}...")

        for epoch in range(epochs):
            self._train_one_epoch(train_loader, optimiser, epoch, verbose)
            self.eval(eval_data)

    def _train_from_qa(self, train_data, eval_data, verbose):
        """
        Train the properties component in _ steps:
            1. Train an autoencoder to encode object images
            2. Cluster the latent space separately for each class (the number of clusters is an estimate based on QA)
            3. Find the optimal property values for each cluster, for each class using ASP
            4. Label the full training dataset using: AE, cluster centres and label->property mapping
            5. Train the property component as if the full dataset was given (ie. same as train_from_hardcoded)

        :param train_data: Training data ((Videos, answers))
        :param eval_data: Eval data (QADataset)
        :param verbose: Verbose printing (bool)
        """

        num_obj = 10000

        videos, answers = train_data
        train_prop_dataset = VideoPropDataset(self.spec, videos, transform=_ae_transform, num_obj_per_cls=num_obj)
        train_prop_qa_dataset = QAPropDataset(self.spec, videos, answers, transform=_ae_transform)

        # Train AE, cluster and find optimal label -> property mappings
        cls_cluster_map = self._find_num_clusters(train_prop_qa_dataset)
        ae_model = self._train_obj_ae(train_prop_dataset, verbose)
        cls_label_centre_map = self._cluster_objects(ae_model, train_prop_dataset, cls_cluster_map)
        cls_label_prop_map = self._find_label_prop_maps(train_prop_qa_dataset, ae_model, cls_label_centre_map)

        # Label data in train_prop_dataset and train NN with labelled data
        train_dataset = self._label_prop_data(train_data, ae_model, cls_label_centre_map, cls_label_prop_map)
        self._train_from_hardcoded(train_dataset, eval_data, verbose)

    def _label_prop_data(self, dataset, ae_model, cls_label_centre_map, cls_label_prop_map):
        """
        Produce a new dataset which the emulates the original but whose objects have all their properties filled

        :param dataset: VideoQA dataset (VideoDataset)
        :param ae_model: Autoencoder model (AutoEncoderModel)
        :param cls_label_centre_map: Dict mapping cls to dict mapping label to centre
        :param cls_label_prop_map: Dict mapping cls to dict mapping label to dict mapping property to value
        :return: VideoDataset object where each object in the Video objects has all properties filled in
        """

        videos = []
        answers = []

        print("Labelling object properties using autoencoder model...")

        cls_obj_map = {cls: [] for cls in self.spec.obj_types()}
        for video_idx in range(len(dataset)):
            video, ans = dataset[video_idx]
            videos.append(video)
            answers.append(ans)
            for frame_idx, frame in enumerate(video.frames):
                for obj in frame.objs:
                    cls_obj_map[obj.cls].append(obj)

        for cls, objs in cls_obj_map.items():
            imgs = [_ae_transform(obj.img) for obj in objs]
            label_centre_map = cls_label_centre_map[cls]
            labels = self._find_labels(imgs, ae_model, label_centre_map)

            for obj_idx, obj in enumerate(objs):
                label = labels[obj_idx]
                props = cls_label_prop_map[cls][label]
                for prop, val in props.items():
                    obj.set_prop_val(prop, val)

        print("Completed labelling.")

        spec = dataset.spec
        hardcoded = dataset.is_hardcoded()
        timing = dataset.detector_timing()
        new_dataset = VideoDataset(spec, videos, answers, timing=timing, hardcoded=hardcoded)
        return new_dataset

    def _train_obj_ae(self, train_dataset, verbose, lr=0.001, batch_size=128, epochs=5):
        """
        Train an autoencoder NN to compress each object image to a 16-d vector

        :param train_dataset: Training data, we only use the objects (VideoPropDataset)
        :param verbose: Verbose printing (bool)
        :return: ObjectAutoEncoder object
        """

        model = ObjectAutoEncoder(self.spec).to(self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_func)
        optimiser = optim.Adam(model.parameters(), lr=lr)

        print(f"Training object autoencoder with device {self.device}...")

        for epoch in range(epochs):
            self._train_one_epoch_ae(model, train_loader, optimiser, epoch, verbose)

        print("Completed object autoencoder training.")

        return model

    def _train_one_epoch_ae(self, model, train_loader, optimiser, epoch, verbose, print_freq=50):
        num_batches = len(train_loader)
        for t, (imgs, _, _) in enumerate(train_loader):
            model.train()

            images = torch.stack(imgs).to(self.device)
            output = model(images).cpu()
            target = images.cpu()

            loss = self._calc_loss_ae(output, target)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % print_freq == 0:
                print(f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.2f}")

    def _calc_loss_ae(self, output, target):
        loss = self._loss_fn_ae(output, target)
        loss = loss.sum(dim=(1, 2, 3))
        loss = loss.mean()
        return loss

    def _cluster_objects(self, ae_model, train_dataset, cls_cluster_map, batch_size=256):
        """
        Find the centre of each cluster (of labels) for each class

        :param ae_model: ObjectAutoEncoder object
        :param train_dataset: Training data, we only use the objects (VideoPropDataset)
        :param cls_cluster_map: Dict from cls to number of clusters ({str: int})
        :param batch_size: Number of elements to pass through AE at a time
        :return: Dict mapping from cls to dict mapping label to cluster centre
        """

        print("Clustering latent vectors of objects in the training data...")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_func)

        latents = []
        classes = []

        ae_model.eval()

        # Apply images to AE
        for imgs, clss, _ in train_loader:
            images = torch.stack(imgs).to(self.device)
            with torch.no_grad():
                obj_latents = list(ae_model.encode(images).cpu().numpy())

            latents.extend(obj_latents)
            classes.extend(clss)

        # Sort by class
        cls_latents = {cls: [] for cls in set(classes)}
        for idx, latent in enumerate(latents):
            cls = classes[idx]
            cls_latents[cls].append(latent)

        # Create a dict mapping from cls to a dict mapping each label to its centre
        cls_label_centre_map = {}
        for cls, latents in cls_latents.items():
            num_clusters = cls_cluster_map[cls]
            clustering = AgglomerativeClustering(n_clusters=num_clusters)
            labels = clustering.fit_predict(latents)

            # Sort latents by label
            label_latents_map = {label: [] for label in set(labels)}
            for idx, label in enumerate(labels):
                latent = latents[idx]
                label_latents_map[label].append(latent)

            # Calc centre for each label
            label_centre_map = {}
            for label, latents_ in label_latents_map.items():
                centre = np.average(latents_, axis=0)
                label_centre_map[label] = centre

            cls_label_centre_map[cls] = label_centre_map

        print("Completed clustering.")

        return cls_label_centre_map

    def _find_num_clusters(self, dataset):
        """
        Calculate the number of expected clusters for each class by looking at the QA pairs
        This is done by listing all possible (ie. in the QA pairs) values of each property

        :param dataset: Data used to calc number of clusters (QAPropDataset)
        :return: Dict from cls to expected number of clusters ({cls (str): num_clusters (int)})
        """

        cls_prop_vals_map = {}

        # Sort data into a set of property values for each property, for each class
        for q_idx in range(len(dataset)):
            imgs, q_cls, q_props = dataset[q_idx]
            cls_props = cls_prop_vals_map.get(q_cls)
            cls_props = {} if cls_props is None else cls_props

            for prop, val in q_props.items():
                val_set = cls_props.get(prop)
                val_set = set() if val_set is None else val_set
                val_set.add(val)
                cls_props[prop] = val_set

            cls_prop_vals_map[q_cls] = cls_props

        cls_cluster_map = {}

        # Find the number of clusters for each class by multiplying the number of property values together
        for cls, prop_vals in cls_prop_vals_map.items():
            num_clusters = 1
            for prop, vals in prop_vals.items():
                num_clusters *= len(vals)

            cls_cluster_map[cls] = num_clusters

        return cls_cluster_map

    def _find_label_prop_maps(self, dataset, ae_model, cls_label_centre_map):
        """
        For each cls find the mapping from labels to property values
        Uses ASP to find the mapping which maximises the number of questions answered correctly

        :param dataset: Training data (QAPropDataset)
        :param ae_model: Autoencoder NN (ObjectAutoEncoder)
        :param cls_label_centre_map: Dict mapping from cls to a dict mapping from label to cluster centre
        :return: Dict mapping from cls to a dict mapping labels to a dict mapping from property to property value
        """

        print("Searching over mappings from cluster labels to property values...")

        cls_asp_data_map = {cls: [] for cls in set(cls_label_centre_map.keys())}
        for cls, label_centre_map in cls_label_centre_map.items():
            for q_idx in range(len(dataset)):
                imgs, q_cls, q_props = dataset[q_idx]
                if cls == q_cls:
                    labels = self._find_labels(imgs, ae_model, label_centre_map)
                    cls_asp_data_map[cls].append((q_idx, q_props, labels))

        cls_label_prop_map = {}
        for cls, data_list in cls_asp_data_map.items():
            q_idxs, q_props, labels = tuple(zip(*data_list))
            asp_str = self._gen_unsup_prop_asp_str(q_idxs, q_props, labels)
            name = "Property component training"
            asp_models = ASPRunner.run(self._temp_asp_file, asp_str, timeout=10, prog_name=name, opt_proven=True)
            label_prop_map = self._parse_asp_models(asp_models, cls)
            cls_label_prop_map[cls] = label_prop_map

        print("Completed label property map search.")

        return cls_label_prop_map

    def _find_labels(self, imgs, ae_model, centres, batch_size=256):
        """
        Find label for each image by calculating latent vector and finding the closest cluster centre
        Will group images into batches if there are than <batch_size> of them

        :param imgs: List of images ([Tensor])
        :param ae_model: Autoencoder model (AutoEncoderModel)
        :param centres: Dict mapping from label to centre np vector
        :param batch_size: Maximum number of images to pass through AE network at once
        :return: List of labels ([int])
        """

        ae_model.eval()
        latents = []

        image_batches = [imgs]
        if len(imgs) > batch_size:
            image_batches = grouper(imgs, batch_size)

        for images in image_batches:
            images = [img for img in images if img is not None]
            images = torch.stack(images).to(self.device)
            with torch.no_grad():
                obj_latents = list(ae_model.encode(images).cpu().numpy())
                latents.extend(obj_latents)

        labels = []
        for latent in latents:
            dists = [(label, cosine(latent, centre)) for label, centre in centres.items()]
            label, _ = min(dists, key=lambda label_dist: label_dist[1])
            labels.append(label)

        return labels

    def _gen_unsup_prop_asp_str(self, q_idxs, prop_vals, labels):
        asp_str = ""

        mapping_str = "{prop}_mapping({label}, {val})"
        answer_head_str = "answer({q_idx}, {prop}, Val)"
        holds_prop_str = "holds({prop}({val}, Id), {q_idx})"
        exp_str = "expected({q_idx}, {prop}, {val})"
        labelled_obj_str = "labelled_obj({id}, {label}, {q_idx})"

        asp_str += "\n% Choice rules for generating possible label to property mappings\n"

        # Generate choice rules for generating all possible mappings
        label_set = set([item for obj_labels in labels for item in obj_labels])
        for label in label_set:
            ans_set_gen_str = ""
            for prop in self.spec.prop_names():
                choice_str = "1 { "
                for val in self.spec.prop_values(prop):
                    val = self.spec.to_internal(prop, val)
                    choice_str += mapping_str.format(prop=prop, label=label, val=val) + " ; "

                ans_set_gen_str += choice_str[:-2] + "} 1.\n"
            asp_str += ans_set_gen_str

        asp_str += "\n% Rules to generate holds() from mappings\n"

        # Generate rules for generating holds() from property mappings
        for prop in self.spec.prop_names():
            rule_str = holds_prop_str.format(prop=prop, val="Val", q_idx="Q") + " :- "
            rule_str += labelled_obj_str.format(id="Id", label="Label", q_idx="Q") + ", "
            rule_str += mapping_str.format(prop=prop, label="Label", val="Val") + ".\n"
            asp_str += rule_str

        asp_str += "\n% Helper rules\n"

        # Generate helper rules
        asp_str += "obj_id(Id, Q) :- labelled_obj(Id, _, Q).\n"
        asp_str += "mapping(Label, Col, Rot) :- colour_mapping(Label, Col), rotation_mapping(Label, Rot).\n"

        asp_str += "\n% Rules and data for each question\n"

        # Some rules and data needs to be generated for each question
        for idx, q_idx in enumerate(q_idxs):
            q_prop_vals = prop_vals[idx]
            obj_labels = labels[idx]

            # Generate answer rules
            for head_prop, exp_val in q_prop_vals.items():
                ans_rule = answer_head_str.format(prop=head_prop, q_idx=q_idx) + " :- "
                for body_prop, body_val in q_prop_vals.items():
                    if head_prop != body_prop:
                        val = self.spec.to_internal(body_prop, body_val)
                        ans_rule += holds_prop_str.format(prop=body_prop, val=val, q_idx=q_idx) + ", "

                ans_rule += holds_prop_str.format(prop=head_prop, val="Val", q_idx=q_idx) + ", "
                ans_rule += f"obj_id(Id, {q_idx}).\n"

                # Expected value
                exp_val = self.spec.to_internal(head_prop, exp_val)
                ans_rule += exp_str.format(prop=head_prop, val=exp_val, q_idx=q_idx) + ".\n"
                asp_str += ans_rule

            asp_str += "\n"

            # Generate labelled object data
            for obj_idx, label in enumerate(obj_labels):
                asp_str += labelled_obj_str.format(id=obj_idx, label=label, q_idx=q_idx) + ".\n"

            asp_str += "\n"

        # Finally, add weak constraint and show commands
        asp_str += "% Optimisation\n"
        asp_str += ":~ answer(Q, Prop, Val), expected(Q, Prop, Val). [-1@1, Q, Prop, Val]\n"
        asp_str += ":~ mapping(Label1, Col, Rot), mapping(Label2, Col, Rot), " \
                   "Label1 != Label2. [1@0, Label1, Label2, Col, Rot]\n\n"

        for prop in self.spec.prop_names():
            asp_str += f"#show {prop}_mapping/2.\n"

        return asp_str

    def _parse_asp_models(self, models, cls):
        """
        Parse the result of the ASP run and construct the label_prop_map for a single class

        :param models: List of list of Symbol objects from clingo API
        :param cls: Class of objects for which the ASP program was run
        :return: Dict mapping labels to a dict mapping from prop to val
        """

        print(f"Found {len(models)} optimal label-property mappings for object type: {cls}")

        # Take a single model since they are all optimal
        model = models[0]

        props = self.spec.prop_names()

        label_prop_map = {}
        for sym in model:
            for prop in props:
                if sym.name == f"{prop}_mapping":
                    label, val = sym.arguments
                    label = label.number
                    val = val.number

                    # Add prop -> val mapping for this label
                    prop_vals = label_prop_map.get(label)
                    prop_vals = {} if prop_vals is None else prop_vals
                    assert prop_vals.get(prop) is None

                    val = self.spec.from_internal(prop, val)
                    prop_vals[prop] = val
                    label_prop_map[label] = prop_vals

        return label_prop_map

    @staticmethod
    def _collate_dicts(dicts):
        props = {}
        for coll in dicts:
            for prop, data in coll.items():
                append_in_map(props, prop, data)

        return props

    def _train_one_epoch(self, train_loader, optimiser, epoch, verbose, print_freq=50):
        num_batches = len(train_loader)
        for t, (imgs, _, objs) in enumerate(train_loader):
            self.model.train()

            images, targets = self._prepare_data(imgs, objs)
            output = self.model(images)
            output = {prop: out.to("cpu") for prop, out in output.items()}
            targets = {prop: target.to("cpu") for prop, target in targets.items()}

            loss, losses = self._calc_loss(output, targets)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            if verbose and (t+1) % print_freq == 0:
                loss_str = f"Epoch {epoch:>3}, batch [{t+1:>4}/{num_batches}] -- overall loss = {loss.item():.6f}"
                for prop, loss in losses.items():
                    loss_str += f" -- {prop} loss = {loss.item():.6f}"
                print(loss_str)

        # Save a temp model every epoch
        current_save = f"{self._temp_save}/after_{epoch + 1}_epochs.pt"
        torch.save(self.model.state_dict(), current_save)

    def _prepare_data(self, imgs, objs, prop=None):
        """
        Prepare the data to be passed to the network

        :param imgs: List of PIL images
        :param objs: List of dicts: {prop: val}
        :param prop: Generate only targets with this property
        :return: 4d torch Tensor, {prop: Tensor}
        """

        targets = {}
        props = self.spec.prop_names() if prop is None else [prop]

        # Convert to tensors
        images = torch.cat([img[None, :, :, :] for img in imgs])
        for prop in props:
            prop_vals = [torch.tensor([self.spec.to_internal(prop, obj[prop])]) for obj in objs]
            targets[prop] = torch.cat(prop_vals)

        # Send to device
        images = images.to(device=self.device)
        targets = {prop: vals.to(self.device) for prop, vals in targets.items()}

        return images, targets

    def _calc_loss(self, preds, targets):
        losses = {prop: self.loss_fn(preds[prop], target) for prop, target in targets.items()}
        loss = sum(losses.values())
        return loss, losses

    @staticmethod
    def _eval_prop(preds, indices, threshold=None):
        """
        Calculate metrics for classification of a single property

        :param preds: Network predictions (tensor of floats, (N, C))
        :param indices: Target class indices (tensor of ints, (N))
        :return: loss (list of floats), TP, FP, TN, FN (all 1d tensors of ints), num_correct (int)
        """

        if threshold is None:
            threshold = 0

        preds_shape = preds.shape
        targets_shape = indices.shape

        assert preds_shape[0] == targets_shape[0], "Predictions and targets must have the same batch size"

        loss = F.cross_entropy(preds, indices, reduction="none")
        loss_batch = list(loss.numpy())

        preds = F.softmax(preds, dim=1)

        # Convert targets to one-hot encoding
        targets = torch.eye(preds_shape[1]).index_select(0, indices)

        act_bool = torch.BoolTensor(targets == 1)
        max_vals, _ = torch.max(preds, 1)
        preds_bool = torch.BoolTensor(preds >= max_vals[:, None])
        preds_bool = preds_bool & (preds >= threshold)

        _, pred_idxs = torch.max(preds_bool, dim=1)
        _, act_idxs = torch.max(act_bool, dim=1)

        num_correct = torch.sum(act_idxs == pred_idxs)
        tps = torch.sum(act_bool & preds_bool, dim=0)
        fps = torch.sum(~act_bool & preds_bool, dim=0)
        tns = torch.sum(~act_bool & ~preds_bool, dim=0)
        fns = torch.sum(act_bool & ~preds_bool, dim=0)

        return loss_batch, tps, fps, tns, fns, num_correct.item()

    def _print_results(self, results, correct, losses, num_predictions):
        tps, fps, tns, fns = results
        for prop in self.spec.prop_names():
            print(f"\nResults for {prop}:")

            loss = losses[prop]
            class_tps = tps[prop]
            class_fps = fps[prop]
            class_tns = tns[prop]
            class_fns = fns[prop]

            print(f"{'Value':<18}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}")

            for val in self.spec.prop_values(prop):
                tp = class_tps[val].item()
                fp = class_fps[val].item()
                tn = class_tns[val].item()
                fn = class_fns[val].item()

                precision = tp / (tp + fp) if (tp + fp) != 0 else float("NaN")
                recall = tp / (tp + fn) if (tp + fn) != 0 else float("NaN")
                accuracy = (tp + tn) / (tp + tn + fp + fn)

                print(f"{val:<18}{accuracy:<12.4f}{precision:<12.4f}{recall:<12.4f}")

            avg_loss = torch.tensor(loss).mean()
            acc = correct[prop] / num_predictions

            print(f"\nAverage loss: {avg_loss:.6f}")
            print(f"Overall accuracy: {acc:.4f}\n")
