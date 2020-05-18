import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from hvqa.events.abs_detector import _AbsEventDetector
from hvqa.util.asp_runner import ASPRunner
from hvqa.util.interfaces import Trainable
from hvqa.util.func import append_in_map


class ILPEventDetector(_AbsEventDetector, Trainable):
    def __init__(self, spec):
        super(ILPEventDetector, self).__init__(spec)

        self.extra_features = [
            ("disappear", self._gen_disappear_features)
        ]

        self.max_rules = 5
        self.hyp_params_str = f"\n#const action={{action}}.\n#const max_rules={self.max_rules}.\n#const fg={{fg}}.\n"

        path = Path("hvqa/events")
        self.asp_data_file = path / "_asp_ilp_data.lp"
        self.asp_opt_file = path / "_asp_opt_file_{action}_{fg}.lp"
        self.background_knowledge_file = path / "occurs_search_bk.lp"
        self.features_file = path / "_features.lp"

        self.feature_str_map = None

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

        feat_str_map, feats_str = self._gen_features()
        self.feature_str_map = feat_str_map
        f = open(self.features_file, "w")
        f.write(feats_str)
        f.close()

        actions = [action for action in self.spec.actions if action != "nothing"]

        num_workers = os.cpu_count()
        executor = ThreadPoolExecutor(max_workers=num_workers)

        start_time = time.time()

        hyp_futures = {}
        for action in actions:
            future = executor.submit(self._find_hypothesis, action)
            future.add_done_callback(lambda fut: print(f"Found hypothesis for {action} action:\n{fut.result()}\n"))
            hyp_futures[action] = future

        hyps = {action: future.result() for action, future in hyp_futures.items()}

        total_time = time.time() - start_time
        print(f"\nCompleted ILP hypothesis search in: {total_time} seconds.\n")

        # Remove temp files
        # self.asp_data_file.unlink()
        # self.features_file.unlink()

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
            action_internal = self.spec.to_internal("action", action)

            opt_str = self.hyp_params_str + "\n"
            opt_str += self._gen_static_predicates()
            opt_str = opt_str.format(action=action_internal, fg=fg)
            opt_str += self._gen_acc_feature_str(acc_features)

            opt_file = str(self.asp_opt_file).format(action=action_internal, fg=fg)
            files = [self.asp_data_file, self.background_knowledge_file, self.features_file]
            prog_name = f"ILP {action} action search with fg={fg}"

            models = ASPRunner.run(opt_file, opt_str, additional_files=files, timeout=3600, prog_name=prog_name)
            completed_fgs.add(fg)

            # Choose a single model to process, since all are considered to be optimal
            acc_features = self._process_opt_model(models[0], completed_fgs)

            curr_hyp = self._gen_hyp(action, acc_features)
            print(f"Found hypothesis for {action} action after optimising feature group {fg}:\n{curr_hyp}\n")

        hyp_str = self._gen_hyp(action, acc_features)
        return hyp_str

    @staticmethod
    def _gen_acc_feature_str(acc_features):
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

    def _gen_hyp(self, action, feature_dicts):
        rule_feature_map = {}
        for feature in feature_dicts:
            append_in_map(rule_feature_map, feature["rule"], feature)

        rules = []
        for _, features in rule_feature_map.items():
            feat_strs = []
            for feature in features:
                fg = feature["fg"]
                f_id = feature["f_id"]
                feat_strs.append(self.feature_str_map[fg][f_id])

            rule_body = ", ".join(feat_strs)
            rule_str = f"occurs({action}(Id), Frame) :- {rule_body}."
            rules.append(rule_str)

        rules_str = "\n\n".join(rules)
        return rules_str

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
                asp_enc = f"actual({action_internal}, {example_num}).\n\n{initial}\n{next_frame}\n"
                examples.append(asp_enc)

                example_num += 1

        return examples

    def _gen_static_predicates(self):
        static_str = "static(Id, I, {is_static}) :- holds(class({cls}, Id), I)."

        statics = []
        for cls in self.spec.obj_types():
            is_static = "true" if self.spec.is_static(cls) else "false"
            statics.append(static_str.format(cls=cls, is_static=is_static))

        statics_str = "\n" + "\n".join(statics) + "\n"
        return statics_str

    def _gen_features(self):
        feature_str_map = {}

        x_pos_feat_map, x_pos_feat_str = self._gen_pos_features("x")
        y_pos_feat_map, y_pos_feat_str = self._gen_pos_features("y")

        feature_str_map["x_pos"] = x_pos_feat_map
        feature_str_map["y_pos"] = y_pos_feat_map

        prop_feats = {prop: self._gen_discrete_prop_features(prop) for prop in self.spec.prop_names()}
        extra_feats = {feat: func() for feat, func in self.extra_features}

        prop_feat_strs = []
        for prop, (feat_map, feat_strs) in prop_feats.items():
            feature_str_map[prop] = feat_map
            prop_feat_strs.append(feat_strs)

        extra_feat_strs = []
        for extra, (feat_map, feat_strs) in extra_feats.items():
            feature_str_map[extra] = feat_map
            extra_feat_strs.append(feat_strs)

        prop_feat_str = "\n\n".join(prop_feat_strs)
        extra_feat_str = "\n\n".join(extra_feat_strs)

        acc_value_str = f"acc_value(Id, Frame, Rule) :- static(Id, Frame, false), " \
                        f"feature_value(x_pos, Id, Frame, Rule), feature_value(y_pos, Id, Frame, Rule)"

        for prop in self.spec.prop_names():
            acc_value_str += f", feature_value({prop}, Id, Frame, Rule)"

        for extra, _ in extra_feats.items():
            acc_value_str += f", feature_value({extra}, Id, Frame, Rule)"

        acc_value_str += "."
        feats_str = f"\n{acc_value_str}\n{x_pos_feat_str}\n{y_pos_feat_str}\n{prop_feat_str}\n{extra_feat_str}\n"

        return feature_str_map, feats_str

    @staticmethod
    def _gen_pos_features(x_y):
        pos_combs = ["{v}1<{v}2", "{v}1>{v}2", "{v}1={v}2", "{v}1!={v}2"]
        rule_str = f"feature_value({x_y}_pos, Id, Frame, Rule) :- {{comb}}, obj_pos((X1, Y1), Id, Frame), " \
                   f"obj_pos((X2, Y2), Id, -Frame), feature({x_y}_pos, {{f_id}}, Rule)."

        asp_var = "X" if x_y == "x" or x_y == "X" else "Y"
        pos_combs = [comb.format(v=asp_var) for comb in pos_combs]

        id_str_map = {}

        pos_rules = []
        feature_weights = []
        for f_id, comb in enumerate(pos_combs):
            pos_rules.append(rule_str.format(comb=comb, f_id=f_id))
            feature_weights.append(f"feature_weight({x_y}_pos, {f_id}, 1).")
            id_str_map[f_id] = f"obj_pos((X1, Y1), Id, Frame), obj_pos((X2, Y2), Id, Frame+1), {comb}"

        empty_id = len(pos_combs)
        empty_rule = f"feature_value({x_y}_pos, Id, Frame, Rule) :- " \
                     f"object(Id, Frame), feature(x_pos, {empty_id}, Rule)."
        pos_rules.append(empty_rule)
        feature_weights.append(f"feature_weight({x_y}_pos, {empty_id}, 0).")
        id_str_map[empty_id] = ""

        asp_str = f"\n% {x_y} position features\n"
        asp_str += "\n\n".join(pos_rules)
        asp_str += "\n\n" + "\n".join(feature_weights)
        asp_str += f"\nempty_id({x_y}_pos, {empty_id}).\n\n"

        return id_str_map, asp_str

    def _gen_discrete_prop_features(self, prop):
        u_prop = prop.capitalize()

        rule_str = f"feature_value({prop}, Id, Frame, Rule) :- {{feature}}, holds({prop}({u_prop}1, Id), Frame), " \
                   f"holds({prop}({u_prop}2, Id), -Frame), feature({prop}, {{f_id}}, Rule)."

        f_id = 0
        asp_rules = []
        feature_weights = []
        id_str_map = {}

        for val1_ in self.spec.prop_values(prop):
            for val2_ in self.spec.prop_values(prop):
                val1 = self.spec.to_internal(prop, val1_)
                val2 = self.spec.to_internal(prop, val2_)
                feature = f"{u_prop}1={val1}, {u_prop}2={val2}"
                asp_rules.append(rule_str.format(feature=feature, f_id=f_id))
                feature_weights.append(f"feature_weight({prop}, {f_id}, 1).")
                id_str_map[f_id] = f"holds({prop}({u_prop}1, Id), Frame), " \
                                   f"holds({prop}({u_prop}2, Id), Frame+1), {feature}"
                f_id += 1

        for feature in [f"{u_prop}1={u_prop}2", f"{u_prop}1!={u_prop}2"]:
            asp_rules.append(rule_str.format(feature=feature, f_id=f_id))
            feature_weights.append(f"feature_weight({prop}, {f_id}, 1).")
            id_str_map[f_id] = f"holds({prop}({u_prop}1, Id), Frame), holds({prop}({u_prop}2, Id), Frame+1), {feature}"
            f_id += 1

        empty_rule = f"feature_value({prop}, Id, Frame, Rule) :- object(Id, Frame), feature({prop}, {f_id}, Rule)."
        asp_rules.append(empty_rule)
        feature_weights.append(f"feature_weight({prop}, {f_id}, 0).")
        id_str_map[f_id] = ""

        asp_str = f"\n% {prop} features\n"
        asp_str += "\n\n".join(asp_rules)
        asp_str += "\n\n" + "\n".join(feature_weights)
        asp_str += f"\nempty_id({prop}, {f_id}).\n\n"

        return id_str_map, asp_str

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

        id_str_map = {
            0: "disappear(Id, Frame+1)",
            1: "object(Id, Frame), not disappear(Id, Frame+1)",
            2: ""
        }

        return id_str_map, asp_str
