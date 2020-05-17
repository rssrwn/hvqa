from pathlib import Path

from hvqa.events.abs_detector import _AbsEventDetector
from hvqa.util.asp_runner import ASPRunner
from hvqa.util.interfaces import Trainable


class ILPEventDetector(_AbsEventDetector, Trainable):
    def __init__(self, spec):
        super(ILPEventDetector, self).__init__(spec)

        self.extra_features = [
            ("disappear", self._gen_disappear_features)
        ]

        self.max_rules = 5
        self.hyp_params_str = f"\n#const action={{action}}.\n#const max_rules={self.max_rules}.\nconst fg={{fg}}.\n"

        path = Path("hvqa/events")
        self.asp_data_file = path / "_asp_ilp_data.lp"
        self.asp_opt_file = path / "_asp_opt_file_{action}.lp"
        self.background_knowledge_file = path / "occurs_search_bk.lp"
        self.features_file = path / "_features.lp"

    @staticmethod
    def new(spec, **kwargs):
        events = ILPEventDetector(spec)
        return events

    @staticmethod
    def load(spec, path):
        pass

    def save(self, path):
        pass

    def _detect_events(self, frames):
        pass

    def train(self, train_data, eval_data, verbose=True):
        """
        Train the event detection component using an ASP ILP search to find rules for each action

        :param train_data: Training data ((videos, answers))
        :param eval_data: Evaluation data
        :param verbose: Print additional info during training
        """

        videos, answers = train_data

        asp_data = self._gen_asp_opt_data(videos, answers)
        f = open(self.asp_data_file, "w")
        f.write("\n".join(asp_data) + "\n\n")
        f.close()

        feats_str = self._gen_features()
        f = open(self.features_file, "w")
        f.write(feats_str)
        f.close()

        for action in self.spec.actions:
            self._find_hypothesis(action)

        # Remove temp files
        self.asp_data_file.unlink()
        self.features_file.unlink()

    def _find_hypothesis(self, action):
        """
        Find hypothesis for an action which explains the observed data

        :param action: Action to find hypothesis for (str)
        :return: String of optimal hypothesis for <action>
        """

        fgs = ["x_pos", "y_pos"] + self.spec.prop_names() + [name for name, _ in self.extra_features]

        completed_fgs = set()
        acc_features = []

        for fg in fgs:
            opt_str = self.hyp_params_str + "\n"
            opt_str += self._gen_static_predicates()
            opt_str = opt_str.format(action=action, fg=fg)

            acc_feature_strs = self._gen_acc_feature_strs(acc_features)
            opt_str += "\n\n" + "\n".join(acc_feature_strs) + "\n\n"

            opt_file = str(self.asp_opt_file).format(action=action)
            files = [self.asp_data_file, self.background_knowledge_file, self.features_file]
            prog_name = f"ILP {action} action search with fg={fg}"

            models = ASPRunner.run(opt_file, opt_str, additional_files=files, timeout=3600, prog_name=prog_name)
            completed_fgs.add(fg)

            # Choose a single model to process, since all are considered to be optimal
            acc_features = self._process_opt_model(models[0], completed_fgs)

        hyp_str = self._gen_hyp(acc_features)
        return hyp_str

    @staticmethod
    def _gen_acc_feature_strs(acc_features):
        acc_feature_str = "acc_feature({fg}, {f_id}, {acc_id})."

        feat_strs = []
        for acc_feat in acc_features:
            fg = acc_feat["fg"]
            f_id = acc_feat["f_id"]
            acc_id = acc_feat["rule"]
            feat_strs.append(acc_feature_str.format(fg=fg, f_id=f_id, acc_id=acc_id))

        feat_str = "\n\n" + "\n".join(feat_strs) + "\n\n"
        return feat_str

    @staticmethod
    def _process_opt_model(model, completed_fgs):
        feature_dicts = []
        for sym in model:
            if sym.name == "feature":
                fg, f_id, rule = sym.arguments
                fg = fg.name
                f_id = f_id.number
                rule = rule.number

                if fg in completed_fgs:
                    feature_dict = {"fg": fg, "f_id": f_id, "rule": rule}
                    feature_dicts.append(feature_dict)

        return feature_dicts

    def _gen_hyp(self, feature_dicts):
        pass

    def _gen_asp_opt_data(self, videos, answers):
        examples = []
        example_num = 1

        for video_idx, video in enumerate(videos):
            q_idxs = [q_idx for q_idx, q_type in enumerate(video.q_types) if q_type == self.spec.qa.event_q]
            for q_idx in q_idxs:
                frame_idx = self.spec.qa.parse_event_question(video.questions[q_idx])
                action = self.spec.qa.parse_event_ans(answers[video_idx][q_idx])
                action_internal = self.spec.to_internal("action", action)

                initial = video.frames[frame_idx].gen_asp_encoding(str(example_num))
                next_frame = video.frames[frame_idx+1].gen_asp_encoding("-" + str(example_num))
                asp_enc = f"actual({action_internal}, {example_num}).\n\n{initial}{next_frame}"
                examples.append(asp_enc)

                example_num += 1

        return examples

    # def _gen_data(self, videos, answers):
    #     action_set = {action for action in self.spec.actions if action != "nothing"}
    #
    #     action_ilasp_enc_map = {action: [] for action in action_set}
    #     action_example_num_map = {action: 0 for action in action_set}
    #
    #     for video_idx, video in enumerate(videos):
    #         q_idxs = [q_idx for q_idx, q_type in enumerate(video.q_types) if q_type == self.spec.qa.event_q]
    #         for q_idx in q_idxs:
    #             frame_idx = self.spec.qa.parse_event_question(video.questions[q_idx])
    #             action = self.spec.qa.parse_event_ans(answers[video_idx][q_idx])
    #
    #             initial = video.frames[frame_idx].gen_asp_encoding("initial_frame")
    #             next_frame = video.frames[frame_idx+1].gen_asp_encoding("next_frame")
    #             example_num = action_example_num_map[action]
    #
    #             # Add the example for each action (making it negative if it is a different action)
    #             for action_ in action_set:
    #                 act_ = self.spec.to_internal("action", action_)
    #                 pos_neg = "pos" if action == action_ else "neg"
    #                 example_name = f"{act_}_p{example_num}" if pos_neg == "pos" else f"{act_}_n{example_num}"
    #                 asp_enc = f"#{pos_neg}({example_name}@1, {{ occurs_{act_} }}, {{}}, {{\n" \
    #                           f"{initial}{next_frame}}}).\n"
    #
    #                 action_ilasp_enc_map[action_].append(asp_enc)
    #             action_example_num_map[action] += 1
    #
    #     return action_ilasp_enc_map

    # def _gen_background_knowledge(self):
    #     back_know = "holds(F, I) :- obs(F, I).\n\n" \
    #                 "step(I) :- obs(_, I).\n\n" \
    #                 "obj_pos((X, Y), Id, I) :- holds(position((X, Y, _, _), Id), I).\n\n" \
    #                 "disappear(Id, I+1) :- \n" \
    #                 "  holds(class(Class, Id), I),\n" \
    #                 "  not holds(class(Class, Id), I+1),\n" \
    #                 "  step(I+1), step(I).\n\n"
    #
    #     # Add static predicate for each class (from spec)
    #     for obj_type in self.spec.obj_types():
    #         static_bool = "true" if self.spec.is_static(obj_type) else "false"
    #         static_rule = f"static(Id, I, {static_bool}) :- holds(class({obj_type}, Id), I).\n"
    #         back_know += static_rule
    #
    #     back_know += "\n"
    #
    #     for action in self.spec.actions:
    #         if action != "nothing":
    #             act = self.spec.to_internal("action", action)
    #             back_know += f"occurs_{act} :- occurs({act}(Id), initial_frame).\n"
    #
    #     back_know += "\n"
    #
    #     # Add rule for nothing action
    #     occurs_nothing = "occurs(nothing(Id), I) :- {neg_actions}static(Id, I, false), step(I+1), step(I)."
    #     action_template = "not occurs({action}(Id), I), "
    #
    #     neg_actions = ""
    #     for action in self.spec.actions:
    #         if action != "nothing":
    #             neg_actions += action_template.format(action=self.spec.to_internal("action", action))
    #
    #     occurs_nothing = occurs_nothing.format(neg_actions=neg_actions)
    #     back_know += occurs_nothing + "\n\n"
    #
    #     return back_know

    # @staticmethod
    # def _extend_rules(rules, ext_strs):
    #     new_rules = []
    #     for ext in ext_strs:
    #         ext_str = ", " + ext if ext != "" else ext
    #         [new_rules.append(rule + ext_str) for rule in rules]
    #
    #     return new_rules

    # def _gen_pos_bias(self, action):
    #     internal_action = self.spec.to_internal("action", action)
    #     rule_start = f"1 ~ occurs({internal_action}(Id), initial_frame) :- " \
    #                  f"static(Id, initial_frame, false), " \
    #                  f"obj_pos((X1, Y1), Id, initial_frame), " \
    #                  f"obj_pos((X2, Y2), Id, next_frame)"
    #
    #     rules = [rule_start]
    #
    #     # Add combinations for xs
    #     exts = [comb.format(p1="X1", p2="X2") for comb in self._pos_combs]
    #     rules = self._extend_rules(rules, exts)
    #
    #     # Add combinations for ys
    #     exts = [comb.format(p1="Y1", p2="Y2") for comb in self._pos_combs]
    #     rules = self._extend_rules(rules, exts)
    #
    #     rules = [rule + "." for rule in rules]
    #     return rules

    # def _gen_bias_rules(self, action):
    #     internal_action = self.spec.to_internal("action", action)
    #     rule_start = f"1 ~ occurs({internal_action}(Id), initial_frame) :- " \
    #                  f"static(Id, initial_frame, false), " \
    #                  f"obj_pos((X1, Y1), Id, initial_frame), " \
    #                  f"obj_pos((X2, Y2), Id, next_frame)"
    #
    #     rules = [rule_start]
    #
    #     # Add combinations for xs
    #     exts = [comb.format(p1="X1", p2="X2") for comb in self._pos_combs]
    #     rules = self._extend_rules(rules, exts)
    #
    #     # Add combinations for ys
    #     exts = [comb.format(p1="Y1", p2="Y2") for comb in self._pos_combs]
    #     rules = self._extend_rules(rules, exts)
    #
    #     # Add combinations for each property
    #     for prop in self.spec.prop_names():
    #         for frame_str in ["initial_frame", "next_frame"]:
    #             holds_prop_str = f"holds({prop}({{val}}, Id), {frame_str})"
    #             prop_vals = [self.spec.to_internal(prop, val) for val in self.spec.prop_values(prop)]
    #             exts = [holds_prop_str.format(val=val) for val in prop_vals]
    #             exts += [""]
    #             rules = self._extend_rules(rules, exts)
    #
    #     rules = [rule + "." for rule in rules]
    #     return rules

    def _gen_static_predicates(self):
        static_str = "static(Id, I, {is_static}) :- holds(class({cls}, Id), I)."

        statics = []
        for cls in self.spec.obj_types():
            is_static = "true" if self.spec.is_static(cls) else "false"
            statics.append(static_str.format(cls=cls, is_static=is_static))

        statics_str = "\n" + "\n".join(statics) + "\n"
        return statics_str

    def _gen_features(self):
        x_pos_feats = self._gen_pos_features("x")
        y_pos_feats = self._gen_pos_features("y")
        prop_feats = "\n\n".join([self._gen_discrete_prop_features(prop) for prop in self.spec.prop_names()])
        extra_feats = "\n\n".join([func() for _, func in self.extra_features])
        feats_str = f"\n{x_pos_feats}\n{y_pos_feats}\n{prop_feats}\n{extra_feats}\n"
        return feats_str

    @staticmethod
    def _gen_pos_features(x_y):
        pos_combs = ["{v}1<{v}2", "{v}1>{v}2", "{v}1={v}2", "{v}1!={v}2"]
        rule_str = f"feature_value({x_y}_pos, Id, Frame, Rule) :- {{comb}}, obj_pos((X1, Y1), Id, Frame), " \
                   f"obj_pos((X2, Y2), Id, -Frame), feature({x_y}_pos, {{f_id}}, Rule)."

        asp_var = "X" if x_y == "x" or x_y == "X" else "Y"
        pos_combs = [comb.format(v=asp_var) for comb in pos_combs]

        pos_rules = []
        feature_weights = []
        for f_id, comb in enumerate(pos_combs):
            pos_rules.append(rule_str.format(comb=comb, f_id=f_id))
            feature_weights.append(f"feature_weight({x_y}_pos, {f_id}, 1).")

        empty_id = str(len(pos_combs))
        empty_rule = f"feature_value({x_y}_pos, Id, Frame, Rule) :- " \
                     f"object(Id, Frame), feature(x_pos, {empty_id}, Rule)."
        pos_rules.append(empty_rule)
        feature_weights.append(f"feature_weight({x_y}_pos, {empty_id}, 0).")

        asp_str = f"\n% {x_y} position features\n"
        asp_str += "\n\n".join(pos_rules)
        asp_str += "\n\n" + "\n".join(feature_weights)
        asp_str += f"\nempty_id({x_y}_pos, {empty_id}).\n\n"

        return asp_str

    def _gen_discrete_prop_features(self, prop):
        rule_str = f"feature_value({prop}, Id, Frame, Rule) :- {{feature}}, holds({prop}(V1, Id), Frame), " \
                   f"holds({prop}(V2, Id), -Frame), feature({prop}, {{f_id}}, Rule)."

        f_id = 0
        asp_rules = []
        feature_weights = []

        for val1 in self.spec.prop_values(prop):
            for val2 in self.spec.prop_values(prop):
                val1 = self.spec.to_internal(prop, val1)
                val2 = self.spec.to_internal(prop, val2)
                feature = f"V1={val1}, V2={val2}"
                asp_rules.append(rule_str.format(feature=feature, f_id=f_id))
                feature_weights.append(f"feature_weight({prop}, {f_id}, 1).")
                f_id += 1

        for feature in ["V1=V2", "V1!=V2"]:
            asp_rules.append(rule_str.format(feature=feature, f_id=f_id))
            feature_weights.append(f"feature_weight({prop}, {f_id}, 0).")
            f_id += 1

        empty_rule = f"feature_value({prop}, Id, Frame, Rule) :- object(Id, Frame), feature({prop}, {f_id}, Rule)."
        asp_rules.append(empty_rule)
        feature_weights.append(f"feature_weight({prop}, {f_id}, 0).")

        asp_str = f"\n% {prop} features\n"
        asp_str += "\n\n".join(asp_rules)
        asp_str += "\n\n" + "\n".join(feature_weights)
        asp_str += f"\nempty_id({prop}, {f_id}).\n\n"

        return asp_str

    @staticmethod
    def _gen_disappear_features():
        dis_str = "feature_value(disappear, Id, Frame, Rule) :- disappear(Id, -Frame), feature(disappear, 0, Rule)."
        not_dis_str = "feature_value(disappear, Id, Frame, Rule) :- object(Id, Frame), " \
                      "not disappear(Id, -Frame), feature(disappear, 1, Rule)."
        empty_dis_str = "feature_value(disappear, Id, Frame, Rule) :- object(Id, Frame), feature(disappear, 2, Rule)."

        weights = [1, 1, 0]
        weight_str = "\n".join([f"feature_weight(disappear, {f_id}, {w})." for f_id, w in enumerate(weights)])

        asp_str = "\n% Disappear features\n"
        asp_str += f"{dis_str}\n\n{not_dis_str}\n\n{empty_dis_str}\n\n"
        asp_str += weight_str
        asp_str += "\n\nempty_id(disappear, 2).\n\n"

        return asp_str
