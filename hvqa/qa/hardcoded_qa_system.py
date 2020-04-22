from pathlib import Path
import clingo

from hvqa.util.interfaces import Component
from hvqa.util.definitions import CLASSES, PROP_LOOKUP
from hvqa.util.func import format_prop_val, format_prop_str, event_to_asp_str, asp_str_to_event, format_occ


class HardcodedASPQASystem(Component):
    def __init__(self, asp_dir):
        path = Path(asp_dir)
        self.qa_system = path / "qa.lp"
        self.features = path / "background_knowledge.lp"
        self._video_info = path / "_temp_video_info.lp"

        asp_template_0 = "answer({{q_idx}}, {prop}, V) :- {asp_obj}, holds({prop}(V, Id), {frame_idx})."

        asp_template_1 = "related({{q_idx}}) :- holds({rel}(Id1, Id2), {frame_idx}), {asp_obj1}, {asp_obj2}. \n" \
                         "answer({{q_idx}}, yes) :- related({{q_idx}}). \n" \
                         "answer({{q_idx}}, no) :- not related({{q_idx}})."

        asp_template_2 = "answer({{q_idx}}, move) :- occurs(move(Id), {frame_idx}). \n" \
                         "answer({{q_idx}}, rotate_left) :- occurs(rotate_left(Id), {frame_idx}). \n" \
                         "answer({{q_idx}}, rotate_right) :- occurs(rotate_right(Id), {frame_idx}). \n" \
                         "answer({{q_idx}}, nothing) :- occurs(nothing(Id), {frame_idx})."

        asp_template_3 = "answer({{q_idx}}, Prop, Before, After) :- \n" \
                         "  changed(Prop, Before, After, Id, {frame_idx}), \n" \
                         "  exists(Id, {frame_idx}+1), \n" \
                         "  {asp_obj}."

        asp_template_4 = "answer({{q_idx}}, N) :- event_count({event}, Id, N), {asp_obj}."

        asp_template_5 = "answer({{q_idx}}, Event) :- event_count(Event, Id, {num}), {asp_obj}."

        asp_template_6 = "answer({{q_idx}}, Action) :- \n" \
                         "  occurs_event(Action, Id, Frame+1), \n" \
                         "  action(Action), \n" \
                         "  event_occurrence({event}, Id, Frame, {occ}), \n" \
                         "  {asp_obj}."

        ans_template_0 = "{prop_val}"
        ans_template_1 = "{ans}"
        ans_template_2 = "{action}"
        ans_template_3 = "Its {prop} changed from {before} to {after}"
        ans_template_4 = "{ans}"
        ans_template_5 = "{event}"
        ans_template_6 = "{action}"

        self.q_funcs = {
            0: (self._parse_q_type_0, asp_template_0),
            1: (self._parse_q_type_1, asp_template_1),
            2: (self._parse_q_type_2, asp_template_2),
            3: (self._parse_q_type_3, asp_template_3),
            4: (self._parse_q_type_4, asp_template_4),
            5: (self._parse_q_type_5, asp_template_5),
            6: (self._parse_q_type_6, asp_template_6)
        }

        self.ans_funcs = {
            0: (self._answer_q_type_0, ans_template_0),
            1: (self._answer_q_type_1, ans_template_1),
            2: (self._answer_q_type_2, ans_template_2),
            3: (self._answer_q_type_3, ans_template_3),
            4: (self._answer_q_type_4, ans_template_4),
            5: (self._answer_q_type_5, ans_template_5),
            6: (self._answer_q_type_6, ans_template_6)
        }

    def run_(self, video):
        answers = self._answer(video)
        video.set_answers(answers)

    def train(self, data):
        pass

    def save(self, path):
        pass

    def _answer(self, video):
        """
        Answer a question on a video

        :param video: Video obj
        :return: Answers: [str]
        """

        questions = video.questions
        q_types = video.q_types

        # Generate ASP encoding for video and questions
        asp_enc = video.gen_asp_encoding()
        for q_idx, question in enumerate(questions):
            q_type = q_types[q_idx]
            q_func, template = self.q_funcs[q_type]
            question_enc = q_func(question, template)
            question_enc = question_enc.format(q_idx=str(q_idx))
            asp_enc += f"\n{question_enc}\n"

        f = open(self._video_info, "w")
        f.write(asp_enc)
        f.close()

        # Add files
        ctl = clingo.Control(message_limit=0)
        ctl.load(str(self.qa_system))
        ctl.load(str(self.features))
        ctl.load(str(self._video_info))

        # Configure the solver
        config = ctl.configuration
        config.solve.models = 0
        config.solve.opt_mode = "optN"

        ctl.ground([("base", [])])

        # Solve AL model with video info
        models = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                models.append(model.symbols(shown=True))

        assert len(models) <= 1, "ASP QA program must contain only a single answer set"

        # If we cannot find any answer return a wrong answer
        if len(models) == 0:
            return "Unknown (unsatisfiable ASP program)"

        model = models[0]

        answers = {q_idx: [] for q_idx in range(len(questions))}
        for sym in model:
            if sym.name == "answer":
                args = sym.arguments
                q_idx = int(args[0].number)
                args = list(map(str, args[1:]))
                answers[q_idx].append(args)

        ans_strs = [None] * len(questions)
        for q_idx, args in answers.items():
            num_ans = len(args)

            if num_ans > 1:
                print(f"WARNING: {num_ans} answers for question {q_idx}. Selecting a single answer...")
                args = [args[0]]

            if num_ans == 0:
                ans_str = "Unknown"
            else:
                q_type = q_types[q_idx]
                q_func, template = self.ans_funcs[q_type]
                ans_str = q_func(args[0], template)

            ans_strs[q_idx] = ans_str

        # Cleanup temp file
        self._video_info.unlink()

        return ans_strs

    def _answer_q_type_0(self, args, template):
        assert len(args) == 2, "Args is not correct length for question type 0"

        prop, prop_val = args

        # TODO Update dataset to use readable version of rotation (eg. upward-facing)
        # prop_val = format_prop_str(prop, prop_val)

        ans_str = template.format(prop_val=prop_val)
        return ans_str

    def _answer_q_type_1(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 1"

        yes_no = args[0]
        ans_str = template.format(ans=yes_no)
        return ans_str

    def _answer_q_type_2(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 2"

        action = args[0]
        if action == "rotate_left":
            action = "rotate left"
        elif action == "rotate_right":
            action = "rotate right"

        ans_str = template.format(action=action)
        return ans_str

    def _answer_q_type_3(self, args, template):
        assert len(args) == 3, "Args is not correct length for question type 3"

        prop, before, after = args
        before = format_prop_str(prop, before)
        after = format_prop_str(prop, after)
        ans_str = template.format(prop=prop, before=before, after=after)
        return ans_str

    def _answer_q_type_4(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 4"

        num = args[0]
        ans_str = template.format(ans=num)
        return ans_str

    def _answer_q_type_5(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 5"

        event = args[0]
        event = asp_str_to_event(event)
        ans_str = template.format(event=event)
        return ans_str

    def _answer_q_type_6(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 6"

        action = args[0]
        action = asp_str_to_event(action)
        ans_str = template.format(action=action)
        return ans_str

    def _parse_q_type_0(self, question, template):
        splits = question.split(" ")
        prop = splits[1]
        frame_idx = int(splits[-1][:-1])

        # Assume only one property value given in question
        cls = splits[4]
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[5]

        asp_obj = self._gen_asp_obj(cls, prop_val, frame_idx, "Id")
        asp_q = template.format(asp_obj=asp_obj, prop=prop, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_1(self, question, template):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        obj1_cls = splits[2]
        obj1_prop_val = None
        rel_idx = 3  # Note: hardcoded for 'close to' other relations will be different TODO
        if obj1_cls not in CLASSES:
            obj1_prop_val = obj1_cls
            obj1_cls = splits[3]
            rel_idx = 4

        rel = splits[rel_idx]

        obj2_cls = splits[rel_idx + 3]
        obj2_prop_val = None
        if obj2_cls not in CLASSES:
            obj2_prop_val = obj2_cls
            obj2_cls = splits[rel_idx + 4]

        asp_obj1 = self._gen_asp_obj(obj1_cls, obj1_prop_val, frame_idx, "Id1")
        asp_obj2 = self._gen_asp_obj(obj2_cls, obj2_prop_val, frame_idx, "Id2")

        asp_q = template.format(rel=rel, asp_obj1=asp_obj1, asp_obj2=asp_obj2, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_2(self, question, template):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])
        asp_q = template.format(frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_3(self, question, template):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        cls = splits[4]
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[5]

        asp_obj = self._gen_asp_obj(cls, prop_val, frame_idx, "Id")
        asp_q = template.format(asp_obj=asp_obj, frame_idx=frame_idx)
        return asp_q

    def _parse_q_type_4(self, question, template):
        splits = question.split(" ")

        cls_idx = 5
        cls = splits[cls_idx]
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls_idx = 6
            cls = splits[cls_idx]

        event = splits[cls_idx + 1:]
        event = " ".join(event)
        event = event[:-1]
        event = event_to_asp_str(event)

        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")
        asp_q = template.format(asp_obj=asp_obj, event=event)
        return asp_q

    def _parse_q_type_5(self, question, template):
        splits = question.split(" ")

        num = int(splits[-2])

        cls = splits[3]
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[4]

        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")
        asp_q = template.format(asp_obj=asp_obj, num=num)
        return asp_q

    def _parse_q_type_6(self, question, template):
        splits = question.split(" ")

        cls = splits[3]
        event_idx = 7
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[4]
            event_idx = 8

        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")

        # Check if there is an occurrence string
        if splits[-1] == "time?":
            occ_str = splits[-2]
            occ = format_occ(occ_str)
            event = splits[event_idx:-4]
        else:
            splits[-1] = splits[-1][:-1]
            event = splits[event_idx:]
            occ = 1

        event = " ".join(event)
        event = event_to_asp_str(event)
        asp_q = template.format(asp_obj=asp_obj, event=event, occ=occ)
        return asp_q

    @staticmethod
    def _gen_asp_obj(cls, prop_val, frame_idx, id_str):
        asp_obj_str = f"holds(class({cls}, {id_str}), {str(frame_idx)})"
        if prop_val is not None:
            prop = PROP_LOOKUP[prop_val]
            prop_val = format_prop_val(prop, prop_val)

            asp_obj_str += f", holds({prop}({prop_val}, {id_str}), {str(frame_idx)})"

        return asp_obj_str
