from pathlib import Path

from hvqa.util.definitions import CLASSES, PROP_LOOKUP
from hvqa.util.exceptions import UnknownQuestionTypeException, UnknownPropertyValueException


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


class HardcodedQASystem(_AbsQASystem):
    def __init__(self, asp_dir):
        path = Path(asp_dir)
        self.qa_system = path / "qa.lp"
        self._video_info = path / "_temp_video_info.lp"
        self._asp_question_templates = {
            0: "answer(V) :- {asp_obj}, holds({prop}(V, Id), {frame_idx}).\n",

            1: "related :- holds({rel}(Id1, Id2), {frame_idx}), {asp_obj1}, {asp_obj2}.\n"
               "answer(yes) :- related.\n"
               "answer(no) :- not related.\n",

            2: "answer(move) :- occurs(move(Id), {frame_idx})."
               "answer(rl) :- occurs(rotate_left(Id), {frame_idx})."
               "answer(rr) :- occurs(rotate_right(Id), {frame_idx}).",

            3: "answer(Prop, Before, After) :- changed(Prop, Before, After, Id, {frame_idx}), "
               "{asp_obj}, exists(Id, {frame_idx}+1).",

            4: "answer(0).",

            5: "answer(move).",

            6: "answer(rotate_left)."
        }

    def answer(self, video, question, q_type):
        asp_enc = video.gen_asp_encoding()
        asp_enc += f"\nquestion_type({q_type}).\n"

        question_enc = self._gen_asp_question(question, q_type)
        asp_enc += f"\n{question_enc}\n"

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
        rel_idx = 3
        if obj1_cls not in CLASSES:
            obj1_prop_val = obj1_cls
            obj1_cls = splits[3]
            rel_idx = 4

        rel = splits[rel_idx]

        obj2_cls = splits[rel_idx + 2]
        obj2_prop_val = None
        if obj2_cls not in CLASSES:
            obj2_prop_val = obj2_cls
            obj2_cls = splits[rel_idx + 3]

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


    @staticmethod
    def _gen_asp_obj(cls, prop_val, frame_idx, id_str):
        asp_obj_str = f"holds(class({cls}, Id), {str(frame_idx)})"
        if prop_val is not None:
            prop = PROP_LOOKUP[prop_val]
            if prop == "rotation":
                if prop_val == "upward-facing":
                    prop_val = 0
                elif prop_val == "right-facing":
                    prop_val = 1
                elif prop_val == "downward-facing":
                    prop_val = 2
                elif prop_val == "left-facing":
                    prop_val = 3
                else:
                    raise UnknownPropertyValueException(f"Rotation {prop_val} unknown")

            asp_obj_str += f", holds({prop}({prop_val}, {id_str}), {str(frame_idx)})"

        return asp_obj_str
