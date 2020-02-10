import torch
import torch.nn.functional as F

from hvqa.util import _AbsEvaluator, get_device


PROPERTIES = ["colour", "rotation", "class"]


class PropertyExtractionEvaluator(_AbsEvaluator):
    """
    Class for evaluating a model's performance at extracting properties
    """

    def eval_model(self, model, threshold=None):
        print("Evaluating property extraction model...")

        device = get_device()
        model.eval()

        # Setup metrics
        losses = [[] for _ in range(len(PROPERTIES))]
        tps = [0 for _ in range(len(PROPERTIES))]
        fps = [0 for _ in range(len(PROPERTIES))]
        tns = [0 for _ in range(len(PROPERTIES))]
        fns = [0 for _ in range(len(PROPERTIES))]

        num_predictions = 0

        for i, (x, y) in enumerate(self.test_loader):
            images = torch.cat([img[None, :, :, :] for img in x])
            targets = [{k: v for k, v in t.items()} for t in y]
            images = images.to(device=device)

            with torch.no_grad():
                colour_preds, rot_preds, class_preds = model(images)

            colour_preds = colour_preds.to("cpu")
            rot_preds = rot_preds.to("cpu")
            class_preds = class_preds.to("cpu")

            colour_targets = torch.cat([target["colour"][None, :] for target in targets]).to("cpu")
            rot_targets = torch.cat([target["rotation"][None, :] for target in targets]).to("cpu")
            class_targets = torch.cat([target["class"][None, :] for target in targets]).to("cpu")

            vals = [
                self._eval_classification(colour_preds, colour_targets, threshold),
                self._eval_classification(rot_preds, rot_targets, threshold),
                self._eval_classification(class_preds, class_targets, threshold)
            ]

            for idx, val in enumerate(vals):
                loss_, tps_, fps_, tns_, fns_ = val
                losses[idx].extend(loss_)
                tps[idx] += tps_
                fps[idx] += fps_
                tns[idx] += tns_
                fns[idx] += fns_

            num_predictions += len(y)

        # Print results
        print(f"{'Property':<12}{'Precision':<12}{'Recall':<12}{'Accuracy':<12}{'Loss':<6}")
        for i in range(len(PROPERTIES)):
            loss = losses[i]
            tp = tps[i]
            fp = fps[i]
            tn = tns[i]
            fn = fns[i]

            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            avg_loss = torch.tensor(loss).mean()

            print(f"{PROPERTIES[i].capitalize():<12}{precision:<12.4f}{recall:<12.4f}{accuracy:<12.4f}{avg_loss:<6.4f}")

    @staticmethod
    def _eval_classification(preds, indices, threshold=None):
        """
        Calculate metrics for classification

        :param preds: Network predictions (tensor of floats, (N, C))
        :param indices: Target class indices (tensor of ints, (N, 1))
        :return: loss (list of floats), TP, FP, TN, FN (all ints)
        """

        if threshold is None:
            threshold = 0

        preds_shape = preds.shape
        targets_shape = indices.shape

        assert preds_shape[0] == targets_shape[0], "Predictions and targets must have the same batch size"

        loss = F.cross_entropy(preds, indices[:, 0], reduction="none")
        loss_batch = list(loss.numpy())

        preds = F.softmax(preds, dim=1)

        # Convert targets to one-hot encoding
        targets = torch.eye(preds_shape[1]).index_select(0, indices[:, 0])

        act_bool = torch.BoolTensor(targets == 1)
        max_vals, _ = torch.max(preds, 1)
        preds_bool = torch.BoolTensor(preds >= max_vals[:, None])
        preds_bool = preds_bool & (preds >= threshold)

        tps = torch.sum(act_bool & preds_bool).item()
        fps = torch.sum(~act_bool & preds_bool).item()
        tns = torch.sum(~act_bool & ~preds_bool).item()
        fns = torch.sum(act_bool & ~preds_bool).item()

        return loss_batch, tps, fps, tns, fns
