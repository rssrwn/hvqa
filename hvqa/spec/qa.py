# Video QA pairs specification class


class QASpec:
    def __init__(self, spec):
        self.env_spec = spec

        self.prop_q = 0
        self.relation_q = 1
        self.event_q = 2

    def parse_prop_question(self, question):
        splits = question.split(" ")
        prop = splits[1]
        frame_idx = int(splits[-1][:-1])

        # Assume only one property value given in question
        cls = splits[4]
        val = None
        if cls not in self.env_spec.obj_types():
            val = cls
            cls = splits[5]

        return prop, val, cls, frame_idx

    def parse_relation_question(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        obj1_cls = splits[2]
        obj1_prop_val = None
        rel_idx = 3  # Note: hardcoded for 'close to' other relations will be different
        if obj1_cls not in self.env_spec.obj_tpyes():
            obj1_prop_val = obj1_cls
            obj1_cls = splits[3]
            rel_idx = 4

        rel = splits[rel_idx]

        obj2_cls = splits[rel_idx + 3]
        obj2_prop_val = None
        if obj2_cls not in self.env_spec.obj_tpyes():
            obj2_prop_val = obj2_cls
            obj2_cls = splits[rel_idx + 4]

        return rel, obj1_cls, obj1_prop_val, obj2_cls, obj2_prop_val, frame_idx
