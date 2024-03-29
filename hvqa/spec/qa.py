# Video QA pairs specification class

from hvqa.util.exceptions import UnknownPropertyValueException, UnknownAnswerException


class QASpec:
    def __init__(self, spec):
        self.spec = spec

        self.prop_q = 0
        self.relation_q = 1
        self.event_q = 2

        self._occurrences = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "fifth": 5
        }

        self._noun_to_event = {
            "moving": "move",
            "rotating left": "rotate left",
            "rotating right": "rotate right",
            "eating a fish": "eat a fish",
            "eating a bag": "eat a bag",
            "changing colour": "change colour"
        }

    def parse_prop_ans(self, ans_str):
        prop_val = ans_str
        prop = self.spec.find_prop(prop_val)
        return prop, prop_val

    def parse_relation_ans(self, ans_str):
        return ans_str

    def parse_event_ans(self, ans_str):
        return ans_str

    def parse_ans_3(self, ans_str):
        splits = ans_str.split(" ")
        prop = splits[1]
        before = splits[4]
        after = splits[6]
        return prop, before, after

    def parse_ans_4(self, ans_str):
        num = int(ans_str)
        return num

    def parse_ans_5(self, ans_str):
        return ans_str

    def parse_ans_6(self, ans_str):
        return ans_str

    def parse_explanation_ans(self, ans_str):
        if ans_str == "The octopus ate the bag":
            ans = 0
        elif ans_str == "The fish was eaten":
            ans = 1
        elif ans_str == "The bag was eaten":
            ans = 2
        else:
            raise UnknownAnswerException(f"The answer {ans_str} was unknown")

        return ans

    def parse_counterfactual_ans(self, ans_str):
        return ans_str

    def parse_prop_question(self, question):
        splits = question.split(" ")
        prop = splits[1]
        frame_idx = int(splits[-1][:-1])

        # Assume only one property value given in question
        cls = splits[4]
        val = None
        if cls not in self.spec.obj_types():
            val = cls
            cls = splits[5]

        return prop, val, cls, frame_idx

    def parse_relation_question(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        obj1_cls = splits[2]
        obj1_prop_val = None
        rel_idx = 3
        if obj1_cls not in self.spec.obj_types():
            obj1_prop_val = obj1_cls
            obj1_cls = splits[3]
            rel_idx = 4

        rel = splits[rel_idx]
        rel_add = 3 if rel == "close" else 2

        obj2_cls = splits[rel_idx + rel_add]
        obj2_prop_val = None
        if obj2_cls not in self.spec.obj_types():
            obj2_prop_val = obj2_cls
            obj2_cls = splits[rel_idx + rel_add + 1]

        return rel, obj1_cls, obj1_prop_val, obj2_cls, obj2_prop_val, frame_idx

    def parse_event_question(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])
        return frame_idx

    def parse_q_3(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        cls = splits[4]
        prop_val = None
        if cls not in self.spec.obj_types():
            prop_val = cls
            cls = splits[5]

        return prop_val, cls, frame_idx

    def parse_q_4(self, question):
        splits = question.split(" ")

        cls_idx = 5
        cls = splits[cls_idx]
        prop_val = None
        if cls not in self.spec.obj_types():
            prop_val = cls
            cls_idx = 6
            cls = splits[cls_idx]

        event = splits[cls_idx + 1:]
        event = " ".join(event)
        event = event[:-1]
        event = self.spec.to_internal("event", event)

        return prop_val, cls, event

    def parse_q_5(self, question):
        splits = question.split(" ")

        num = int(splits[-2])

        cls = splits[3]
        prop_val = None
        if cls not in self.spec.obj_types():
            prop_val = cls
            cls = splits[4]

        return num, prop_val, cls

    def parse_q_6(self, question):
        splits = question.split(" ")

        cls = splits[3]
        event_idx = 7
        prop_val = None
        if cls not in self.spec.obj_types():
            prop_val = cls
            cls = splits[4]
            event_idx = 8

        # Check if there is an occurrence string
        if splits[-1] == "time?":
            occ_str = splits[-2]
            occ = self._format_occ(occ_str)
            event = splits[event_idx:-4]
        else:
            splits[-1] = splits[-1][:-1]
            event = splits[event_idx:]
            occ = 1

        event = " ".join(event)
        event = self._format_event_noun(event)
        event = self.spec.to_internal("event", event)

        return prop_val, cls, occ, event

    def parse_explanation_question(self, question):
        splits = question.split(" ")
        rotation = splits[3]
        return rotation

    def parse_counterfactual_question(self, question):
        splits = question.split(" ")
        rock_colour = splits[-2]
        return rock_colour

    def _format_event_noun(self, event_noun):
        event = self._noun_to_event[event_noun]
        return event

    def _format_occ(self, occ_str):
        occ = self._occurrences.get(occ_str)
        if occ is None:
            raise UnknownPropertyValueException(f"Unknown occurrence value {occ_str}")

        return occ
