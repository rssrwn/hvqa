# File for modelling the environment

from hvqa.util.func import collect_obj


class Obj:
    def __init__(self, spec, cls, pos):
        """
        Create an object from a video

        :param spec: EnvSpec object
        :param cls: Class of object (str)
        :param pos: Position of object (4-tuple of int)
        """

        assert cls in spec.obj_types(), f"Class must be one of {spec.obj_types()}"

        self.spec = spec

        self.cls = cls
        self.pos = pos
        self.is_static = spec.is_static(cls)
        self.id = None
        self.img = None
        self.prop_vals = {prop: None for prop in spec.prop_names()}

    def set_id(self, id_):
        self.id = id_

    def set_prop_val(self, prop, val):
        vals = self.spec.prop_values(prop)
        assert prop in self.spec.prop_names(), f"Unknown property {prop}"
        assert val in vals, f"{prop} of value {val} must be one of {vals}"

        self.prop_vals[prop] = val

    def set_image(self, img):
        """
        Set the img param for this object by cropping the frame with object position

        :param img: Full frame
        """

        assert self.pos is not None, "Position must be set in order to extract object image"
        self.img = collect_obj(img, self.pos)

    def gen_asp_encoding(self, frame_num, body=None):
        assert self.cls is not None, "Class must be set"
        assert self.pos is not None, "Position must be set"
        assert self.id is not None, "Id must be set"
        for prop, val in self.prop_vals.items():
            assert val is not None, f"{prop} must be set"

        self.pos = tuple(map(int, self.pos))

        body_str = "" if body is None else " :- " + body
        frame_num = str(frame_num)
        encoding = f"obs(class({self.cls}, {self.id}), {frame_num}){body_str}.\n" \
                   f"obs(position({str(self.pos)}, {self.id}), {frame_num}){body_str}.\n"

        for prop, val in self.prop_vals.items():
            asp_val = str(self.spec.to_internal(prop, val))
            encoding += f"obs({prop}({asp_val}, {self.id}), {frame_num}){body_str}.\n"

        return encoding


class Frame:
    def __init__(self, spec, objs):
        self.spec = spec
        self.objs = objs
        self.relations = []
        self._id_idx_map = {}
        self._try_id_idx_map = {}

    def set_obj_ids(self, ids):
        assert len(self.objs) == len(ids), "Number of ids must the same as the number of objects"
        for id_, obj in zip(ids, self.objs):
            obj.set_id(id_)

        # Find which objs are duplicate for err_corr
        self._id_idx_map = self._find_duplicate_idxs()

    def set_relation(self, idx1, idx2, relation):
        assert relation in self.spec.relations, f"Relation {relation} must be one of {self.spec.relations}"

        id1 = self.objs[idx1].id
        id2 = self.objs[idx2].id
        self.relations.append((id1, id2, relation))

    def gen_asp_encoding(self, frame_num):
        enc = ""

        try_id = 0
        try_id_idx_map = {}
        idx_try_id_map = {}
        for id_, idxs in self._id_idx_map.items():
            choice_str = "1 {"
            for idx in idxs:
                choice_str += f" try_obj({try_id}, {frame_num}) ;"
                try_id_idx_map[try_id] = idx
                idx_try_id_map[idx] = try_id
                try_id += 1

            choice_str = choice_str[:-1] + "} 1.\n"
            enc += choice_str

        self._try_id_idx_map = try_id_idx_map

        for idx, obj in enumerate(self.objs):
            err_id = idx_try_id_map.get(idx)
            if err_id is not None:
                body_str = f"try_obj({err_id}, {frame_num})"
                enc += obj.gen_asp_encoding(frame_num, body_str) + "\n"
            else:
                enc += obj.gen_asp_encoding(frame_num) + "\n"

        for id1, id2, rel in self.relations:
            enc += f"obs({rel}({str(id1)}, {str(id2)}), {str(frame_num)}).\n"

        enc += "\n"

        return enc

    def _find_duplicate_idxs(self):
        """
        For each id where there is uncertainty about which object is correct
        Finds the list of indices (into the objs list) competing for the id

        :return: Map from id to list of indices into objs list
        """

        ids = {}
        for idx, obj in enumerate(self.objs):
            idxs = ids.get(obj.id)
            idxs = [] if idxs is None else idxs
            idxs.append(idx)
            ids[obj.id] = idxs

        dup_ids = {id_: idxs for id_, idxs in ids.items() if len(idxs) > 1}
        return dup_ids

    def set_correct_objs(self, try_ids):
        """
        Remove error objects from frame
        <try_ids> are the correct identifiers set using a choice rule in the ASP encoding

        :param try_ids: List of identifiers
        """

        remove_idxs = set()
        for try_id in try_ids:
            obj_idx = self._try_id_idx_map[try_id]
            obj_id = self.objs[obj_idx].id
            dup_idxs = self._id_idx_map[obj_id]
            remove_idxs = remove_idxs.union(set([idx for idx in dup_idxs if idx != obj_idx]))

        objs = [obj for idx, obj in enumerate(self.objs) if idx not in remove_idxs]

        self.objs = objs
        self._id_idx_map = self._find_duplicate_idxs()
        self._id_idx_map = {}
        self._try_id_idx_map = {}


class Video:
    def __init__(self, spec, frames):
        self.spec = spec
        self.frames = frames
        self.actions = [[]] * (spec.num_frames - 1)
        self.questions = None
        self.q_types = None
        self.answers = None

    def add_action(self, action, obj_id, start_idx):
        assert action in self.spec.actions, f"Action {action} is not one of {self.spec.actions}"
        self.actions[start_idx] = self.actions[start_idx] + [(action, obj_id)]

    def set_questions(self, questions, q_types):
        self.questions = questions
        self.q_types = q_types

    def set_answers(self, answers):
        self.answers = answers

    def gen_asp_encoding(self):
        enc = ""
        for frame_idx, frame in enumerate(self.frames):
            enc += frame.gen_asp_encoding(frame_idx)

        enc += "\n"

        for frame_idx, events in enumerate(self.actions):
            for event, obj_id in events:
                enc += f"occurs({event}({obj_id}), {frame_idx}).\n"

        return enc
