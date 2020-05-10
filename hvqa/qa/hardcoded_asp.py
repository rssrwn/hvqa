from pathlib import Path
import clingo

from hvqa.util.interfaces import Component


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
        prop, prop_val = self.spec.qa.parse_ans_0(args)
        ans_str = template.format(prop_val=prop_val)
        return ans_str

    def _answer_q_type_1(self, args, template):
        yes_no = self.spec.qa.parse_ans_1(args)
        ans_str = template.format(ans=yes_no)
        return ans_str

    def _answer_q_type_2(self, args, template):
        action = self.spec.qa.parse_ans_2(args)
        ans_str = template.format(action=action)
        return ans_str

    def _answer_q_type_3(self, args, template):
        prop, before, after = self.spec.qa.parse_ans_3(args)
        ans_str = template.format(prop=prop, before=before, after=after)
        return ans_str

    def _answer_q_type_4(self, args, template):
        num = self.spec.qa.parse_ans_4(args)
        ans_str = template.format(ans=num)
        return ans_str

    def _answer_q_type_5(self, args, template):
        event = self.spec.qa.parse_ans_5(args)
        ans_str = template.format(event=event)
        return ans_str

    def _answer_q_type_6(self, args, template):
        action = self.spec.qa.parse_ans_6(args)
        ans_str = template.format(action=action)
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
        frame_idx = self.spec.qa.parse_q_2(question)
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

    def _gen_asp_obj(self, cls, prop_val, frame_idx, id_str):
        asp_obj_str = f"holds(class({cls}, {id_str}), {str(frame_idx)})"
        if prop_val is not None:
            prop = self.spec.find_prop(prop_val)
            prop_val = self.spec.to_internal(prop, prop_val)
            asp_obj_str += f", holds({prop}({prop_val}, {id_str}), {str(frame_idx)})"

        return asp_obj_str
