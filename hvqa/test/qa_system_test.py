import unittest

from hvqa.qa.hardcoded_qa_system import HardcodedASPQASystem
from hvqa.util.video_repr import Video, Frame, Obj


q_type_0 = "What colour was the left-facing rock in frame 3?"
q_type_1 = "Was the octopus close to the blue rock in frame 12?"
q_type_2 = "Which action occurred immediately after frame 17?"
q_type_3 = "What happened to the octopus immediately after frame 23?"

args_type_0 = ("rotation", "1")
args_type_1 = ("yes",)
args_type_2 = ("rotate_left",)
args_type_3 = ("colour", "blue", "purple")


obj1 = Obj("octopus", (10, 50, 27, 67))
obj1.colour = "red"
obj1.rot = 1
obj1.id = 0

obj2 = Obj("octopus", (25, 50, 42, 67))
obj2.colour = "blue"
obj2.rot = 1
obj2.id = 0

obj3 = Obj("octopus", (25, 50, 42, 67))
obj3.colour = "blue"
obj3.rot = 2
obj3.id = 0

obj_rock = Obj("rock", (30, 70, 35, 75))
obj_rock.colour = "blue"
obj_rock.rot = 0
obj_rock.id = 1

obj_fish = Obj("fish", (200, 100, 210, 110))
obj_fish.colour = "silver"
obj_fish.rot = 3
obj_fish.id = 2

frame1 = Frame([obj1, obj_rock, obj_fish])
frame2 = Frame([obj2, obj_rock, obj_fish])
frame2.set_relation(0, 1, "close")
frame3 = Frame([obj3, obj_rock, obj_fish])
frame3.set_relation(0, 1, "close")

frames1 = [frame1, frame2, frame3]
video1 = Video(frames1)
video1.add_event("move", 0, 0)
video1.add_event("rotate_right", 0, 1)

e2e_q_type_0 = "What colour was the octopus in frame 1?"
e2e_q_type_1 = "Was the octopus close to the blue rock in frame 1?"
e2e_q_type_2 = "Which action occurred immediately after frame 1?"
e2e_q_type_3 = "What happened to the octopus immediately after frame 0?"


class QASystemTest(unittest.TestCase):
    def setUp(self):
        self.qa = HardcodedASPQASystem("hvqa/qa")

    def test_gen_asp_question_type_0(self):
        expected = "answer(colour, V) :- holds(class(rock, Id), 3), " \
                   "holds(rotation(3, Id), 3), holds(colour(V, Id), 3).\n"

        asp = self.qa._gen_asp_question(q_type_0, 0)
        self.assertEqual(expected, asp)

    def test_gen_asp_question_type_1(self):
        expected = "related :- holds(close(Id1, Id2), 12), holds(class(octopus, Id1), 12), " \
                   "holds(class(rock, Id2), 12), holds(colour(blue, Id2), 12).\n"\
                   "answer(yes) :- related.\n"\
                   "answer(no) :- not related.\n"

        asp = self.qa._gen_asp_question(q_type_1, 1)
        self.assertEqual(expected, asp)

    def test_gen_asp_question_type_2(self):
        expected = "answer(move) :- occurs(move(Id), 17).\n"\
                   "answer(rotate_left) :- occurs(rotate_left(Id), 17).\n"\
                   "answer(rotate_right) :- occurs(rotate_right(Id), 17).\n"

        asp = self.qa._gen_asp_question(q_type_2, 2)
        self.assertEqual(expected, asp)

    def test_gen_asp_question_type_3(self):
        expected = "answer(Prop, Before, After) :- " \
                   "changed(Prop, Before, After, Id, 23), holds(class(octopus, Id), 23), exists(Id, 23+1).\n"

        asp = self.qa._gen_asp_question(q_type_3, 3)
        self.assertEqual(expected, asp)

    def test_gen_answer_type_0(self):
        expected = "right-facing"
        answer = self.qa._gen_answer_str(args_type_0, 0)
        self.assertEqual(expected, answer)

    def test_gen_answer_type_1(self):
        expected = "yes"
        answer = self.qa._gen_answer_str(args_type_1, 1)
        self.assertEqual(expected, answer)

    def test_gen_answer_type_2(self):
        expected = "rotate left"
        answer = self.qa._gen_answer_str(args_type_2, 2)
        self.assertEqual(expected, answer)

    def test_gen_answer_type_3(self):
        expected = "Its colour changed from blue to purple"
        answer = self.qa._gen_answer_str(args_type_3, 3)
        self.assertEqual(expected, answer)

    def test_e2e_answer_type_0(self):
        expected = "blue"
        ans = self.qa.answer(video1, e2e_q_type_0, 0)
        self.assertEqual(expected, ans)

    def test_e2e_answer_type_1(self):
        expected = "yes"
        ans = self.qa.answer(video1, e2e_q_type_1, 1)
        self.assertEqual(expected, ans)

    def test_e2e_answer_type_2(self):
        expected = "rotate right"
        ans = self.qa.answer(video1, e2e_q_type_2, 2)
        self.assertEqual(expected, ans)

    def test_e2e_answer_type_3(self):
        expected = "Its colour changed from red to blue"
        ans = self.qa.answer(video1, e2e_q_type_3, 3)
        self.assertEqual(expected, ans)
