from pathlib import Path
from clyngor import solve

from hvqa.util.definitions import CLASSES, PROP_LOOKUP
from hvqa.util.exceptions import UnknownQuestionTypeException
from hvqa.util.func import format_prop_val


class _AbsQASystem:
    def answer(self, video, question, q_type):
        """
        Answer a question on a video

        :param video: Video obj
        :param question: Question: str
        :param q_type: Question type: int
        :return: Answer: str
        """

        raise NotImplementedError


class HardcodedASPQASystem(_AbsQASystem):
    def __init__(self, asp_dir):
        path = Path(asp_dir)
        self.qa_system = path / "qa.lp"
        self.features = path / "background_knowledge.lp"
        self._video_info = path / "_temp_video_info.lp"

        self._asp_question_templates = {
            0: "answer({prop}, V) :- {asp_obj}, holds({prop}(V, Id), {frame_idx}).\n",

            1: "related :- holds({rel}(Id1, Id2), {frame_idx}), {asp_obj1}, {asp_obj2}.\n"
               "answer(yes) :- related.\n"
               "answer(no) :- not related.\n",

            2: "answer(move) :- occurs(move(Id), {frame_idx})."
               "answer(rotate_left) :- occurs(rotate_left(Id), {frame_idx})."
               "answer(rotate_right) :- occurs(rotate_right(Id), {frame_idx}).",

            3: "answer(Prop, Before, After) :- changed(Prop, Before, After, Id, {frame_idx}), "
               "{asp_obj}, exists(Id, {frame_idx}+1).",

            4: "answer(0).",

            5: "answer(move).",

            6: "answer(rotate_left)."
        }

        self._answer_str_templates = {
            0: "{prop_val}",
            1: "{ans}",
            2: "{action}",
            3: "Its {prop} changed from {before} to {after}",
            4: "not implemented",
            5: "not implemented",
            6: "not implemented"
        }

    def answer(self, video, question, q_type):
        asp_enc = video.gen_asp_encoding()
        asp_enc += f"\nquestion_type({q_type}).\n"

        question_enc = self._gen_asp_question(question, q_type)
        asp_enc += f"\n{question_enc}\n"

        f = open(self._video_info, "w")
        f.write(asp_enc)
        f.close()

        # Solve AL model with video info
        answers = solve([self.features, self.qa_system, self._video_info], use_clingo_module=True)

        assert len(answers) != 0, "ASP QA program is unsatisfiable"
        assert not len(answers) > 1, "ASP QA program must contain only a single answer set"

        answers = answers[0]

        ans_str = None
        for pred, args in answers:
            if pred == "answer":
                ans_str = self._gen_answer_str(args, q_type)

        # Cleanup temp file
        self._video_info.unlink()

        assert ans_str is not None, "The answer set did not contain an answer predicate"

        return ans_str

    def _gen_answer_str(self, args, q_type):
        if q_type == 0:
            ans = self._answer_q_type_0(args)
        elif q_type == 1:
            ans = self._answer_q_type_1(args)
        elif q_type == 2:
            ans = self._answer_q_type_2(args)
        elif q_type == 3:
            ans = self._answer_q_type_3(args)
        elif q_type == 4:
            ans = self._answer_q_type_4(args)
        elif q_type == 5:
            ans = self._answer_q_type_5(args)
        elif q_type == 6:
            ans = self._answer_q_type_6(args)
        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return ans

    def _answer_q_type_0(self, args):
        assert len(args) == 2, "Args is not correct length for question type 0"

        prop, prop_val = args
        prop_val = format_prop_val(prop, prop_val)

        template = self._answer_str_templates[0]
        ans_str = template.format(prop_val=prop_val)
        return ans_str

    def _answer_q_type_1(self, args):
        assert len(args) == 1, "Args is not correct length for question type 1"

        yes_no = args[0]

        template = self._answer_str_templates[1]
        ans_str = template.format(ans=yes_no)
        return ans_str

    def _answer_q_type_2(self, args):
        assert len(args) == 1, "Args is not correct length for question type 2"

        action = args[0]
        if action == "rotate_left":
            action = "rotate left"
        elif action == "rotate_right":
            action = "rotate right"

        template = self._answer_str_templates[2]
        ans_str = template.format(action=action)
        return ans_str

    def _answer_q_type_3(self, args):
        assert len(args) == 3, "Args is not correct length for question type 3"

        prop, before, after = args

        template = self._answer_str_templates[3]
        ans_str = template.format(prop=prop, before=before, after=after)
        return ans_str

    def _answer_q_type_4(self, args):
        assert len(args) == 1, "Args is not correct length for question type 4"

        template = self._answer_str_templates[4]
        return template

    def _answer_q_type_5(self, args):
        assert len(args) == 1, "Args is not correct length for question type 5"

        template = self._answer_str_templates[5]
        return template

    def _answer_q_type_6(self, args):
        assert len(args) == 1, "Args is not correct length for question type 6"

        template = self._answer_str_templates[6]
        return template

    def _gen_asp_question(self, question, q_type):
        if q_type == 0:
            asp_q = self._parse_q_type_0(question)
        elif q_type == 1:
            asp_q = self._parse_q_type_1(question)
        elif q_type == 2:
            asp_q = self._parse_q_type_2(question)
        elif q_type == 3:
            asp_q = self._parse_q_type_3(question)
        elif q_type == 4:
            asp_q = self._parse_q_type_4(question)
        elif q_type == 5:
            asp_q = self._parse_q_type_5(question)
        elif q_type == 6:
            asp_q = self._parse_q_type_6(question)
        else:
            raise UnknownQuestionTypeException(f"Question type {q_type} unknown")

        return asp_q

    def _parse_q_type_0(self, question):
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

        template = self._asp_question_templates[0]
        asp_q = template.format(asp_obj=asp_obj, prop=prop, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_1(self, question):
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

        template = self._asp_question_templates[1]
        asp_q = template.format(rel=rel, asp_obj1=asp_obj1, asp_obj2=asp_obj2, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_2(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        template = self._asp_question_templates[2]
        asp_q = template.format(frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_3(self, question):
        splits = question.split(" ")
        frame_idx = int(splits[-1][:-1])

        cls = splits[4]
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[5]

        asp_obj = self._gen_asp_obj(cls, prop_val, frame_idx, "Id")

        template = self._asp_question_templates[3]
        asp_q = template.format(asp_obj=asp_obj, frame_idx=frame_idx)
        return asp_q

    def _parse_q_type_4(self, question):
        template = self._asp_question_templates[4]
        return template

    def _parse_q_type_5(self, question):
        template = self._asp_question_templates[5]
        return template

    def _parse_q_type_6(self, question):
        template = self._asp_question_templates[6]
        return template

    @staticmethod
    def _gen_asp_obj(cls, prop_val, frame_idx, id_str):
        asp_obj_str = f"holds(class({cls}, {id_str}), {str(frame_idx)})"
        if prop_val is not None:
            prop = PROP_LOOKUP[prop_val]
            prop_val = format_prop_val(prop, prop_val)

            asp_obj_str += f", holds({prop}({prop_val}, {id_str}), {str(frame_idx)})"

        return asp_obj_str
