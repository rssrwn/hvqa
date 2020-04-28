from hvqa.util.definitions import CLOSE_TO, RELATIONS
from hvqa.util.interfaces import Component


class HardcodedRelationClassifier(Component):
    def __init__(self):
        super(HardcodedRelationClassifier, self).__init__()

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

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError()

    @staticmethod
    def new(spec, **kwargs):
        relations = HardcodedRelationClassifier()
        return relations

    @staticmethod
    def load(path):
        raise NotImplementedError("HardcodedRelationClassifier objects must be newly created")

    def save(self, path):
        pass

    def _check_related(self, obj1, obj2):
        obj_relations = []
        for idx, relation_func in enumerate(self.relation_funcs):
            related = relation_func(obj1, obj2)
            if related:
                obj_relations.append(RELATIONS[idx])

        return obj_relations

    @staticmethod
    def _close_to(obj1, obj2):
        """
        Returns whether the obj1 is close to the obj2
        A border is created around the obj1, obj2 is close if it is within the border

        :param obj1: Object 1
        :param obj2: Object 2
        :return: bool
        """

        obj1_x1, obj1_y1, obj1_x2, obj1_y2 = obj1.pos
        obj1_x1 -= CLOSE_TO
        obj1_x2 += CLOSE_TO
        obj1_y1 -= CLOSE_TO
        obj1_y2 += CLOSE_TO

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