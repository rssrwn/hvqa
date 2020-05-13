from hvqa.events.abs_asp import _AbsEventDetector
from hvqa.util.interfaces import Trainable


class ILASPEventDetector(_AbsEventDetector, Trainable):
    def __init__(self, spec):
        super(ILASPEventDetector, self).__init__(spec)

        self.pos_combs = ["{p1}<{p2}", "{p1}>{p2}", "{p1}={p2}", ""]
        self.background_knowledge = self._gen_background_knowledge()
        self.show_occurs = "\n#show occurs/2.\n"

    @staticmethod
    def new(spec, **kwargs):
        events = ILASPEventDetector(spec)
        return events

    @staticmethod
    def load(spec, path):
        pass

    def save(self, path):
        pass

    def _detect_events(self, frames):
        pass

    def train(self, train_data, eval_data, verbose=True):
        pass

    def _gen_background_knowledge(self):
        back_know = "holds(F, I) :- obs(F, I).\n\n" \
                    "step(I) :- obs(_, I).\n\n" \
                    "obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).\n\n" \
                    "disappear(Id, I+1) :- \n" \
                    "  holds(class(Class, Id), I),\n" \
                    "  not holds(class(Class, Id), I+1),\n" \
                    "  step(I+1), step(I).\n\n"

        # Add static predicate for each class (from spec)
        for obj_type in self.spec.obj_types():
            static_bool = "true" if self.spec.is_static(obj_type) else "false"
            static_rule = f"static(Id, I, {static_bool}) :- holds(class({obj_type}, Id), I)."
            back_know += static_rule

        # Add rule for nothing action
        occurs_nothing = "occurs(nothing(Id), I) :- {neg_actions}static(Id, I, false), step(I+1), step(I)."
        action_template = "not occurs({action}(Id), I), "

        neg_actions = ""
        for action in self.spec.actions:
            if action != "nothing":
                neg_actions += action_template.format(action=action)

        occurs_nothing = occurs_nothing.format(neg_actions=neg_actions)
        back_know += occurs_nothing

        return back_know

    @staticmethod
    def _extend_rules(rules, ext_strs):
        new_rules = []
        for ext in ext_strs:
            ext_str = ",\n  " + ext if ext != "" else ext
            [new_rules.append(rule + ext_str) for rule in rules]

        return new_rules

    def _gen_bias_rules(self, action):
        rule_start = f"1 ~ occurs({action}(Id), initial_frame) :- \n" \
                     f"  static(Id, initial_frame, False),\n" \
                     f"  obj_pos((X1, Y1), Id, initial_frame),\n" \
                     f"  obj_pos((X2, Y2), Id, next_frame)"

        rules = [rule_start]

        # Add combinations for xs
        exts = [comb.format(p1="X1", p2="X2") for comb in self.pos_combs]
        rules = self._extend_rules(rules, exts)

        # Add combinations for ys
        exts = [comb.format(p1="Y1", p2="Y2") for comb in self.pos_combs]
        rules = self._extend_rules(rules, exts)

        # Add combinations for each property
        for prop in self.spec.prop_names():
            for frame_str in ["initial_frame", "next_frame"]:
                holds_prop_str = f"holds({prop}({{val}}, Id), {frame_str})"
                exts = [holds_prop_str.format(val=val) for val in self.spec.prop_values(prop)]
                exts += [""]
                rules = self._extend_rules(rules, exts)

        rules = [rule + "." for rule in rules]
        return rules
