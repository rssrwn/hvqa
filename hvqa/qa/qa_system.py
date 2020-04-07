from pathlib import Path

from hvqa.util.definitions import CLASSES, PROP_LOOKUP
from hvqa.util.exceptions import UnknownQuestionTypeException


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
            0: "answer(V) :- {asp_obj_id}, holds({prop}(V, Id), {frame_idx}).\n",

            1: "related :- holds({rel}({obj1_id}, {obj2_id}), {frame_idx}), {asp_obj1_id}, {asp_obj2_id}.\n"
               "answer(yes) :- related.\n"
               "answer(no) :- not related.\n",

            2: "answer(move) :- occurs(move(Id), {frame_idx})."
               "answer(rl) :- occurs(rotate_left(Id), {frame_idx})."
               "answer(rr) :- occurs(rotate_right(Id), {frame_idx}).",

            3: "answer(Prop, Before, After) :- changed(Prop, Before, After, Id, {frame_idx}), "
               "{asp_obj_id}, exists(Id, {frame_idx}+1).",

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
        cls = splits[4]

        # Assume only one property value given in question
        prop_val = None
        if cls not in CLASSES:
            prop_val = cls
            cls = splits[5]

        asp_obj_id = self._gen_asp_obj_id(cls, prop_val, frame_idx)

        template = self._asp_question_templates[0]
        asp_q = template.format(asp_obj_id=asp_obj_id, prop=prop, frame_idx=str(frame_idx))
        return asp_q

    def _parse_q_type_1(self, question):
        pass

    def _gen_asp_obj_id(self, cls, prop_val, frame_idx):
        asp_obj_str = f"holds(class({cls}, Id), {str(frame_idx)})"
        if prop_val is not None:
            prop = PROP_LOOKUP[prop_val]
            asp_obj_str += f", holds({prop}({prop_val}, Id), {str(frame_idx)})"

        return asp_obj_str
