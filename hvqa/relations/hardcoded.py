import torch

from hvqa.util.interfaces import Component
from hvqa.relations.dataset import QARelationDataset


class HardcodedRelationClassifier(Component):
    def __init__(self, spec):
        super(HardcodedRelationClassifier, self).__init__()

        self.spec = spec
        self.close_to_def = 5
        self.relation_funcs = [self._close_to]

    def run_(self, video):
        for frame in video.frames:
            objs = frame.objs
            rels = self._detect_relations(objs)
            for idx1, idx2, rel in rels:
                frame.set_relation(idx1, idx2, rel)

    def _detect_relations(self, objs):
        """
        Detect binary relations between pairs of objects

        :param objs: List of Obj objects
        :return: List of relations [(idx1, idx2, relation)]
        """

        rels = []
        for obj1_idx, obj1 in enumerate(objs):
            for obj2_idx, obj2 in enumerate(objs):
                if obj1_idx != obj2_idx:
                    obj_rels = self._check_related(obj1, obj2)
                    obj_rels = [(obj1_idx, obj2_idx, rel) for rel in obj_rels]
                    rels.extend(obj_rels)

        return rels

    @staticmethod
    def new(spec, **kwargs):
        relations = HardcodedRelationClassifier(spec)
        return relations

    def eval(self, eval_data):
        qa_data = QARelationDataset.from_video_dataset(self.spec, eval_data, enc_objs=False)
        rel_metrics = self._eval_dataset(qa_data)
        self._print_results(rel_metrics)

    def _eval_dataset(self, eval_data):
        rel_preds = {rel: [] for rel in self.spec.relations}
        rel_acts = {rel: [] for rel in self.spec.relations}
        for idx in range(len(eval_data)):
            for rel, (objs, related) in eval_data[idx].items():
                obj1, obj2 = objs
                rels = self._check_related(obj1, obj2)
                pred = True if rel in rels else False
                rel_preds[rel].append(pred)
                rel_acts[rel].append(related)

        rel_metrics = {}
        for rel, preds in rel_preds.items():
            acts = rel_acts[rel]
            preds_bool = torch.BoolTensor(preds)
            acts_bools = torch.BoolTensor(acts)
            num_correct = torch.sum(preds_bool == acts_bools).item()
            tp = torch.sum(preds_bool & acts_bools).item()
            tn = torch.sum(~preds_bool & ~acts_bools).item()
            fp = torch.sum(preds_bool & ~acts_bools).item()
            fn = torch.sum(~preds_bool & acts_bools).item()

            assert num_correct == tp + tn

            rel_metrics[rel] = (tp, tn, fp, fn)

        return rel_metrics

    def _print_results(self, rel_metrics):
        total = 0
        correct = 0
        for rel, (tp, tn, fp, fn) in rel_metrics.items():
            correct_rel = tp + tn
            total_rel = tp + tn + fp + fn
            correct += correct_rel
            total += total_rel

            tp_perc = f"{tp / total_rel:.2f}"
            tn_perc = f"{tn / total_rel:.2f}"
            fp_perc = f"{fp / total_rel:.2f}"
            fn_perc = f"{fn / total_rel:.2f}"

            print(f"\nConfusion matrix for {rel}:")
            print(f"{'Actual:':<10}")
            print(f"{'Pred:':>10}{'Yes':^10}{'No':^10}")
            print(f"{'Yes':<10}{'TP ' + tp_perc:^10}{'FN ' + fn_perc:^10}")
            print(f"{'No':<10}{'FP ' + fp_perc:^10}{'TN ' + tn_perc:^10}")

            acc = correct_rel / total_rel
            precision = tp / (tp + fp) if (tp + fp) != 0 else "NaN"
            recall = tp / (tp + fn) if (tp + fn) != 0 else "NaN"
            f1 = (2 * precision * recall) / (precision + recall) if precision != "NaN" and recall != "NaN" else "NaN"

            precision = f"{precision:.2f}" if type(precision) == float else precision
            recall = f"{recall:.2f}" if type(recall) == float else recall
            f1 = f"{f1:.2f}" if type(f1) == float else f1

            print(f"\nResults for {rel}:")
            print(f"{'Accuracy:':<10} {acc:.2f}")
            print(f"{'Precision:':<10} {precision}")
            print(f"{'Recall:':<10} {recall}")
            print(f"{'F1 Score:':<10} {f1}")

        overall_acc = correct / total
        print(f"\nOverall accuracy: {overall_acc:.2f}\n")

    def _check_related(self, obj1, obj2):
        obj_relations = []
        for idx, relation_func in enumerate(self.relation_funcs):
            related = relation_func(obj1, obj2)
            if related:
                obj_relations.append(self.spec.relations[idx])

        return obj_relations

    def _close_to(self, obj1, obj2):
        """
        Returns whether the obj1 is close to the obj2
        A border is created around the obj1, obj2 is close if it is within the border

        :param obj1: Object 1
        :param obj2: Object 2
        :return: bool
        """

        obj1_x1, obj1_y1, obj1_x2, obj1_y2 = obj1.pos
        obj1_x1 -= self.close_to_def
        obj1_x2 += self.close_to_def
        obj1_y1 -= self.close_to_def
        obj1_y2 += self.close_to_def

        x1, y1, x2, y2 = obj2.pos
        obj_corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

        x_overlap = x2 >= obj1_x2 and x1 <= obj1_x1
        y_overlap = y2 >= obj1_y2 and y1 <= obj1_y1

        for x, y in obj_corners:
            match_x = obj1_x1 <= x <= obj1_x2 or x_overlap
            match_y = obj1_y1 <= y <= obj1_y2 or y_overlap
            if match_x and match_y:
                return True

        return False
