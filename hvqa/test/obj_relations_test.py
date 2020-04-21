import unittest

from hvqa.util.video_repr import Obj
from hvqa.relations.hardcoded_relations import HardcodedRelationClassifier


obj1 = Obj("octopus", (10, 100, 20, 110))
obj2 = Obj("fish", (22, 113, 30, 140))
obj3 = Obj("rock", (22, 200, 30, 210))
obj4 = Obj("bag", (25, 115, 200, 250))
obj5 = Obj("fish", (100, 200, 110, 210))


class ObjRelationsTest(unittest.TestCase):
    def setUp(self):
        self.relation_classifier = HardcodedRelationClassifier()

    def test_is_close_to(self):
        is_close = self.relation_classifier._close_to(obj1, obj2)
        self.assertTrue(is_close)

    def test_is_close_borderline(self):
        is_close = self.relation_classifier._close_to(obj1, obj4)
        self.assertTrue(is_close)

    def test_not_is_close_to(self):
        is_close = self.relation_classifier._close_to(obj1, obj3)
        self.assertFalse(is_close)

    def test_is_close_no_overlap(self):
        is_close = self.relation_classifier._close_to(obj1, obj5)
        self.assertFalse(is_close)

    def test_close_relation_no_overlap(self):
        objs = [obj1, obj2, obj3]
        relations = self.relation_classifier.detect_relations(objs)

        self.assertTrue((0, 1, "close") in relations)
        self.assertTrue((1, 0, "close") in relations)

        close_relations = [(idx1, idx2, rel) for idx1, idx2, rel in relations if rel == "close"]
        exp_length = 2

        self.assertEqual(exp_length, len(close_relations))

    def test_close_relation_overlap(self):
        objs = [obj1, obj4, obj5]
        relations = self.relation_classifier.detect_relations(objs)

        self.assertTrue((0, 1, "close") in relations)
        self.assertTrue((1, 0, "close") in relations)
        self.assertTrue((1, 2, "close") in relations)
        self.assertTrue((2, 1, "close") in relations)

        close_relations = [(idx1, idx2, rel) for idx1, idx2, rel in relations if rel == "close"]
        exp_length = 4

        self.assertEqual(exp_length, len(close_relations))
