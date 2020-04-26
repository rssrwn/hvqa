# File for modelling the environment

from hvqa.util.func import collect_obj


class EnvSpec:
    def __init__(self, num_frames, obj_types, properties, relations, actions, events):
        """
        A class for storing a specification of an environment
        This specification is used to create each Component in a VideoQA Model

        :param num_frames: Number of frames in the video (int)
        :param obj_types: Object classes and is_static (list of (str, bool))
        :param properties: Dict of non-obj detection properties and their values (dict from str to list of str)
        :param relations: Binary relations between objects (list of str)
        :param actions: Actions for objects (list of str)
        :param events: Effects of actions (list of str)

        Note: Properties should not include class or position, these are implicit
        """

        self.num_frames = num_frames
        self._obj_types = [cls for cls, _ in obj_types]
        self._static = [is_static for _, is_static in obj_types]
        self._obj_idx_map = {obj: idx for idx, obj in enumerate(self._obj_types)}
        self._props = properties
        self._prop_names = list(self._props.keys())
        self.relations = relations
        self.actions = actions
        self.events = events
        self._prop_idx_map = {prop: idx for idx, prop in enumerate(self._prop_names)}
        self._val_to_internal_map, self._internal_to_val_map = self._setup_prop_val_maps()

    def _setup_prop_val_maps(self):
        val_to_int = {}
        int_to_val = {}
        for prop in self.prop_names():
            val_to_int[prop] = {val: idx for idx, val in enumerate(self.prop_values(prop))}
            int_to_val[prop] = {idx: val for idx, val in enumerate(self.prop_values(prop))}

        return val_to_int, int_to_val

    @staticmethod
    def from_dict(coll):
        num_frames = coll["num_frames"]
        obj_types = coll["obj_types"]
        properties = coll["properties"]
        relations = coll["relations"]
        actions = coll["actions"]
        events = coll["events"]

        spec = EnvSpec(num_frames, obj_types, properties, relations, actions, events)
        return spec

    def num_props(self):
        return len(self._props.keys())

    def prop_values(self, prop):
        return self._props[prop]

    def to_internal(self, prop, val):
        return self._val_to_internal_map[prop][val]

    def from_internal(self, prop, val):
        return self._internal_to_val_map[prop][val]

    def prop_names(self):
        return self._prop_names

    def obj_types(self):
        return self._obj_types

    def is_static(self, cls):
        cls_idx = self._obj_idx_map[cls]
        is_static = self._static[cls_idx]
        return is_static


class Obj:
    def __init__(self, spec, cls, pos):
        """
        Create an object from a video

        :param spec: EnvSpec object
        :param cls: Class of object (str)
        :param pos: Position of object (4-tuple of int)
        """

        assert cls in spec.obj_types(), f"Class must be one of {spec.obj_types}"

        self.spec = spec

        self.cls = cls
        self.pos = pos
        self.is_static = spec.is_static(cls)
        self.id = None
        self.img = None
        self.prop_vals = {prop: None for prop in spec.prop_names()}

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
        for prop, val in self.prop_vals:
            assert val is not None, f"{prop} must be set"

        self.pos = tuple(map(int, self.pos))

        body_str = "" if body is None else " :- " + body
        frame_num = str(frame_num)
        encoding = f"obs(class({self.cls}, {self.id}), {frame_num}){body_str}.\n" \
                   f"obs(position({str(self.pos)}, {self.id}), {frame_num}){body_str}.\n"

        for prop, val in self.prop_vals:
            encoding += f"obs({prop}({str(val)}, {self.id}), {frame_num}){body_str}.\n"

        return encoding


class Frame:
    def __init__(self, spec, img):
        self.spec = spec
        self.img = img
        self.objs = None
        self._id_idx_map = None
        self.relations = []
        self._id_idx_map = {}
        self._try_id_idx_map = {}

    def set_objs(self, objs):
        self.objs = objs
        self._id_idx_map = self._find_duplicate_idxs()

    def set_relation(self, idx1, idx2, relation):
        assert relation in self.spec.relations, f"Relation arg must be one of {self.spec.relations}"

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
