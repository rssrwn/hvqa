# Video QA pairs specification class

from hvqa.util.exceptions import UnknownPropertyValueException


class QASpec:
    def __init__(self, spec):
        self.spec = spec

        self.prop_q = 0
        self.relation_q = 1
        self.event_q = 2

        # TODO Fix firth
        self._occurrences = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
            "firth": 5
        }

    def parse_ans_0(self, args):
        assert len(args) == 2, "Args is not correct length for question type 0"
        prop, prop_val = args

        # TODO Update dataset to use readable version of rotation (eg. upward-facing)
        if prop != "rotation":
            prop_val = self.spec.from_internal(prop, int(prop_val))

        return prop, prop_val

    def parse_ans_1(self, args):
        assert len(args) == 1, "Args is not correct length for question type 1"
        yes_no = args[0]
        return yes_no

    def parse_ans_2(self, args):
        assert len(args) == 1, "Args is not correct length for question type 2"

        action = args[0]
        if action == "rotate_left":
            action = "rotate left"
        elif action == "rotate_right":
            action = "rotate right"

        return action

    def parse_ans_3(self, args):
        assert len(args) == 3, "Args is not correct length for question type 3"

        prop, before, after = args
        before = self.spec.from_internal(prop, int(before))
        after = self.spec.from_internal(prop, int(after))
        return prop, before, after

    def parse_ans_4(self, args):
        assert len(args) == 1, "Args is not correct length for question type 4"
        num = args[0]
        return num

    def parse_ans_5(self, args):
        assert len(args) == 1, "Args is not correct length for question type 5"
        event = args[0]
        event = self.spec.from_internal(event)
        return event

    def parse_ans_6(self, args):
        assert len(args) == 1, "Args is not correct length for question type 6"
        action = args[0]
        action = self.spec.from_internal(action)
        return action

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
        rel_idx = 3  # Note: hardcoded for 'close to' other relations will be different
        if obj1_cls not in self.spec.obj_tpyes():
            obj1_prop_val = obj1_cls
            obj1_cls = splits[3]
            rel_idx = 4

        rel = splits[rel_idx]

        obj2_cls = splits[rel_idx + 3]
        obj2_prop_val = None
        if obj2_cls not in self.spec.obj_tpyes():
            obj2_prop_val = obj2_cls
            obj2_cls = splits[rel_idx + 4]

        return rel, obj1_cls, obj1_prop_val, obj2_cls, obj2_prop_val, frame_idx

    def parse_q_2(self, question):
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
        event = self.spec.to_internal(event)

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
        event = self.spec.to_internal(event)

        return prop_val, cls, occ, event

    def _format_occ(self, occ_str):
        occ = self._occurrences.get(occ_str)
        if occ is None:
            raise UnknownPropertyValueException(f"Unknown occurrence value {occ_str}")

        return occ
