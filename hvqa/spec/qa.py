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
