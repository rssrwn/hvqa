import unittest

from hvqa.qa.qa_system import HardcodedASPQASystem


q_type_0 = "What colour was the left-facing rock in frame 3?"
q_type_1 = "Was the octopus close to the blue rock in frame 12?"
q_type_2 = "Which action occurred immediately after frame 17?"


class QASystemTest(unittest.TestCase):
    def setUp(self):
        self.qa = HardcodedASPQASystem("hvqa/qa")

    def test_gen_asp_question_type_0(self):
        expected = "answer(colour, V) :- holds(class(rock, Id), 3), holds(rotation(3, Id), 3), holds(colour(V, Id), 3).\n"
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
        expected = ""
