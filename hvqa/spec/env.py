# Video environment specification class

from hvqa.spec.qa import QASpec


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

        # Public members
        self.num_frames = num_frames
        self.relations = relations
        self.actions = actions
        self.events = events

        # Property helpers
        self._props = properties
        self._prop_names = list(self._props.keys())

        # Object helpers
        self._obj_types = [cls for cls, _ in obj_types]
        self._static = [is_static for _, is_static in obj_types]

        # Helper maps (must be done after properties and objects)
        self._val_to_prop_map = self._setup_val_to_prop_map()
        self._obj_idx_map = {obj: idx for idx, obj in enumerate(self._obj_types)}
        self._prop_idx_map = {prop: idx for idx, prop in enumerate(self._prop_names)}
        self._val_to_internal_map, self._internal_to_val_map = self._setup_prop_val_maps()

        self.qa = QASpec(self)

    # TODO Currently assumes that each val is unique
    def _setup_val_to_prop_map(self):
        val_to_prop = {}
        for prop, vals in self._props.items():
            for val in vals:
                val_to_prop[val] = prop

        return val_to_prop

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

    def to_dict(self):
        obj_types = list(zip(self._obj_types, self._static))
        spec = {
            "num_frames": self.num_frames,
            "obj_types": obj_types,
            "properties": self._props,
            "relations": self.relations,
            "actions": self.actions,
            "events": self.events
        }
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

    def find_prop(self, val):
        """
        Find which property a particular property value belongs to
        Note: Currently returns a single property if a value can be in two properties

        :param val: Property value
        :return: Property
        """

        return self._val_to_prop_map[val]
