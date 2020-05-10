# Video QA pairs specification class

from hvqa.util.exceptions import UnknownPropertyValueException


class QASpec:
    def __init__(self, spec):
        self.env_spec = spec

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

    def parse_q_2(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])
        return frame_idx

    def parse_q_3(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        cls = splits[4]
        prop_val = None
        if cls not in self.env_spec.obj_types():
            prop_val = cls
            cls = splits[5]

        return prop_val, cls, frame_idx

    def parse_q_4(self, question):
        splits = question.split(" ")

        cls_idx = 5
        cls = splits[cls_idx]
        prop_val = None
        if cls not in self.env_spec.obj_types():
            prop_val = cls
            cls_idx = 6
            cls = splits[cls_idx]

        event = splits[cls_idx + 1:]
        event = " ".join(event)
        event = event[:-1]
        event = event_to_asp_str(event)

        return prop_val, cls, event

    def parse_q_5(self, question):
        splits = question.split(" ")

        num = int(splits[-2])

        cls = splits[3]
        prop_val = None
        if cls not in self.env_spec.obj_types():
            prop_val = cls
            cls = splits[4]

        return num, prop_val, cls

    def parse_q_6(self, question):
        splits = question.split(" ")

        cls = splits[3]
        event_idx = 7
        prop_val = None
        if cls not in self.env_spec.obj_types():
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
        event = event_to_asp_str(event)
        
        return prop_val, cls, occ, event

    def _event_to_asp_str(event):
        if event == "move" or event == "moving":
            asp_str = "move"
        elif event == "rotate left" or event == "rotating left":
            asp_str = "rotate_left"
        elif event == "rotate right" or event == "rotating right":
            asp_str = "rotate_right"
        elif event == "change colour" or event == "changing colour":
            asp_str = "change_colour"
        elif event == "eat a fish" or event == "eating a fish":
            asp_str = "eat_fish"
        elif event == "eat a bag" or event == "eating a bag":
            asp_str = "eat_bag"
        else:
            raise UnknownPropertyValueException(f"Unknown event {event}")

        return asp_str

    def _asp_str_to_event(asp_event):
        if asp_event == "move":
            event = "move"
        elif asp_event == "rotate_left":
            event = "rotate left"
        elif asp_event == "rotate_right":
            event = "rotate right"
        elif asp_event == "change_colour":
            event = "change colour"
        elif asp_event == "eat_fish":
            event = "eat a fish"
        elif asp_event == "eat_bag":
            event = "eat a bag"
        elif asp_event == "unknown":
            event = "unknown"
        else:
            raise UnknownPropertyValueException(f"Unknown ASP event {asp_event}")

        return event

    def _format_occ(self, occ_str):
        occ = self._occurrences.get(occ_str)
        if occ is None:
            raise UnknownPropertyValueException(f"Unknown occurrence value {occ_str}")

        return occ
