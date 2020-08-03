from pathlib import Path

from hvqa.util.interfaces import Component
from hvqa.util.asp_runner import ASPRunner


class HardcodedASPQASystem(Component):
    def __init__(self, spec):
        self.spec = spec

        path = Path("hvqa/qa")
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

        asp_template_7 = "answer({{q_idx}}, 0) :- disappear_rot_cls({rotation}, octopus). \n" \
                         "answer({{q_idx}}, 1) :- disappear_rot_cls({rotation}, fish). \n" \
                         "answer({{q_idx}}, 2) :- disappear_rot_cls({rotation}, bag). \n"

        asp_template_8 = "octo_col_wo_rock(Colour, I) :- octo_colour(Colour, I), Colour != {colour}. \n" \
                         "later(C1, C2) :- octo_col_wo_rock(C1, I1), octo_col_wo_rock(C2, I2), I1 > I2. \n" \
                         "answer({{q_idx}}, C2) :- \n" \
                         "  octo_col_wo_rock(C1, _), \n" \
                         "  octo_col_wo_rock(C2, _), \n" \
                         "  not later(C1, C2) "

        ans_template_0 = "{prop_val}"
        ans_template_1 = "{ans}"
        ans_template_2 = "{action}"
        ans_template_3 = "Its {prop} changed from {before} to {after}"
        ans_template_4 = "{ans}"
        ans_template_5 = "{event}"
        ans_template_6 = "{action}"
        ans_template_7 = "{ans}"
        ans_template_8 = "{colour}"

        self.q_funcs = {
            0: (self._parse_q_type_0, asp_template_0),
            1: (self._parse_q_type_1, asp_template_1),
            2: (self._parse_q_type_2, asp_template_2),
            3: (self._parse_q_type_3, asp_template_3),
            4: (self._parse_q_type_4, asp_template_4),
            5: (self._parse_q_type_5, asp_template_5),
            6: (self._parse_q_type_6, asp_template_6),
            7: (self._parse_q_type_7, asp_template_7),
            8: (self._parse_q_type_8, asp_template_8)
        }

        self.ans_funcs = {
            0: (self._answer_q_type_0, ans_template_0),
            1: (self._answer_q_type_1, ans_template_1),
            2: (self._answer_q_type_2, ans_template_2),
            3: (self._answer_q_type_3, ans_template_3),
            4: (self._answer_q_type_4, ans_template_4),
            5: (self._answer_q_type_5, ans_template_5),
            6: (self._answer_q_type_6, ans_template_6),
            7: (self._answer_q_type_7, ans_template_7),
            8: (self._answer_q_type_8, ans_template_8)
        }

    def run_(self, video):
        answers = self._answer(video)
        video.set_answers(answers)

    @staticmethod
    def new(spec, **kwargs):
        qa = HardcodedASPQASystem(spec)
        return qa

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

        name = "QA component"
        files = [self.qa_system, self.features]
        models = ASPRunner.run(self._video_info, asp_enc, additional_files=files, prog_name=name, opt_proven=False)

        if len(models) > 1:
            print("WARNING: Multiple answer sets for QA component")

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

        return ans_strs

    def _answer_q_type_0(self, args, template):
        assert len(args) == 2, "Args is not correct length for question type 0"

        prop, prop_val = args
        prop_val = self.spec.from_internal(prop, int(prop_val))

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
        before = self.spec.from_internal(prop, int(before))
        after = self.spec.from_internal(prop, int(after))
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
        event = self.spec.from_internal("event", event)
        ans_str = template.format(event=event)
        return ans_str

    def _answer_q_type_6(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 6"

        action = args[0]
        action = self.spec.from_internal("action", action)
        ans_str = template.format(action=action)
        return ans_str

    def _answer_q_type_7(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 7"

        ans_num = args[0]
        if ans_num == 0:
            ans_str = "The octopus ate a bag"
        elif ans_num == 1:
            ans_str = "The fish was eaten"
        elif ans_num == 2:
            ans_str = "The bag was eaten"
        else:
            ans_str = "Unknown"

        return ans_str

    def _answer_q_type_8(self, args, template):
        assert len(args) == 1, "Args is not correct length for question type 8"

        colour = args[0]
        ans_str = template.format(colour=colour)
        return ans_str

    def _parse_q_type_0(self, question, template):
        prop, prop_val, cls, frame_idx = self.spec.qa.parse_prop_question(question)
        asp_obj = self._gen_asp_obj(cls, prop_val, frame_idx, "Id")
        asp_q = template.format(asp_obj=asp_obj, prop=prop, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_1(self, question, template):
        rel, obj1_cls, obj1_val, obj2_cls, obj2_val, frame_idx = self.spec.qa.parse_relation_question(question)
        asp_obj1 = self._gen_asp_obj(obj1_cls, obj1_val, frame_idx, "Id1")
        asp_obj2 = self._gen_asp_obj(obj2_cls, obj2_val, frame_idx, "Id2")
        asp_q = template.format(rel=rel, asp_obj1=asp_obj1, asp_obj2=asp_obj2, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_2(self, question, template):
        frame_idx = self.spec.qa.parse_event_question(question)
        asp_q = template.format(frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_3(self, question, template):
        prop_val, cls, frame_idx = self.spec.qa.parse_q_3(question)
        asp_obj = self._gen_asp_obj(cls, prop_val, frame_idx, "Id")
        asp_q = template.format(asp_obj=asp_obj, frame_idx=frame_idx)
        return asp_q

    def _parse_q_type_4(self, question, template):
        prop_val, cls, event = self.spec.qa.parse_q_4(question)
        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")
        asp_q = template.format(asp_obj=asp_obj, event=event)
        return asp_q

    def _parse_q_type_5(self, question, template):
        num, prop_val, cls = self.spec.qa.parse_q_5(question)
        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")
        asp_q = template.format(asp_obj=asp_obj, num=num)
        return asp_q

    def _parse_q_type_6(self, question, template):
        prop_val, cls, occ, event = self.spec.qa.parse_q_6(question)
        asp_obj = self._gen_asp_obj(cls, prop_val, "Frame", "Id")
        asp_q = template.format(asp_obj=asp_obj, event=event, occ=occ)
        return asp_q

    def _parse_q_type_7(self, question, template):
        rotation = self.spec.qa.parse_explanation_question(question)
        asp_q = template.format(rotation=rotation)
        return asp_q

    def _parse_q_type_8(self, question, template):
        rock_colour = self.spec.qa.parse_counterfactual_question(question)
        asp_q = template.format(colour=rock_colour)
        return asp_q

    def _gen_asp_obj(self, cls, prop_val, frame_idx, id_str):
        asp_obj_str = f"holds(class({cls}, {id_str}), {str(frame_idx)})"
        if prop_val is not None:
            prop = self.spec.find_prop(prop_val)
            prop_val = self.spec.to_internal(prop, prop_val)
            asp_obj_str += f", holds({prop}({prop_val}, {id_str}), {str(frame_idx)})"

        return asp_obj_str
