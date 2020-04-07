import unittest

from hvqa.util.video_repr import Obj
from hvqa.tracking.obj_tracker import ObjTracker


obj1 = Obj("rock", (30, 30, 60, 60))
obj1_copy = Obj("rock", (30, 30, 60, 60))
obj1_similar = Obj("rock", (35, 35, 65, 65))
obj2 = Obj("fish", (100, 30, 200, 200))
obj3 = Obj("octopus", (110, 40, 0, 0))
obj4 = Obj("fish", (30, 30, 60, 60))
obj5 = Obj("rock", (110, 40, 0, 0))


class ObjTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjTracker()

    def test_dist_is_zero(self):
        dist = self.tracker.dist(obj1, obj1)
        self.assertEqual(dist, 0, "Distance should be zero")

    def test_dist_is_correct(self):
        exp_dist = 70
        dist = self.tracker.dist(obj1, obj2)
        self.assertEqual(exp_dist, dist, "Incorrect distance")

    def test_close_obj_true(self):
        is_close = self.tracker.close_obj(obj2, obj3)
        self.assertTrue(is_close, "A difference of 10 or less should be close")

    def test_close_obj_false(self):
        is_close = self.tracker.close_obj(obj1, obj3)
        self.assertFalse(is_close, "A difference of more than 10 should not be close")

    def test_find_best_match(self):
        objs = enumerate([obj1_similar, obj1_copy, obj2, obj3])
        exp_idx = 1
        idx = self.tracker._find_best_match(obj1, objs)
        self.assertEqual(exp_idx, idx, "Found wrong match for obj")

    def test_find_best_match_moved(self):
        objs = enumerate([obj2, obj3, obj1_similar])
        exp_idx = 2
        idx = self.tracker._find_best_match(obj1, objs)
        self.assertEqual(exp_idx, idx, "Found wrong match after obj moved")

    def test_find_best_match_no_class_match(self):
        objs = enumerate([obj2, obj3, obj4])
        exp_idx = None
        idx = self.tracker._find_best_match(obj1, objs)
        self.assertEqual(exp_idx, idx, "Expected no match (expected 'None' returned)")

    def test_find_best_match_no_pos_match(self):
        objs = enumerate([obj2, obj3, obj5])
        exp_idx = None
        idx = self.tracker._find_best_match(obj1, objs)
        self.assertEqual(exp_idx, idx, "Expected no match (expected 'None' returned)")

    def test_process_frame_initial_frame(self):
        objs = [obj1, obj2, obj3, obj4, obj5]
        exp_idxs = [0, 1, 2, 3, 4]
        idxs = self.tracker.process_frame(objs)
        self.assertEqual(exp_idxs, idxs)

    def test_process_frame_no_change(self):
        objs = [obj1, obj2, obj3, obj4, obj5]
        exp_idxs = [0, 1, 2, 3, 4]
        for _ in range(2):
            idxs = self.tracker.process_frame(objs)
            self.assertEqual(exp_idxs, idxs)

    def test_process_frame_moved_obj(self):
        objs1 = [obj2, obj3, obj4, obj5, obj1]
        objs2 = [obj2, obj3, obj1_similar, obj4, obj5]
        exp_idxs = [[0, 1, 2, 3, 4], [0, 1, 4, 2, 3]]
        idxs = self._process_frames([objs1, objs2])
        self.assertEqual(exp_idxs, idxs)

    def test_process_frame_new_obj(self):
        objs1 = [obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj1, obj5]
        exp_idxs = [[0, 1, 2, 3], [0, 1, 2, 4, 3]]
        idxs = self._process_frames([objs1, objs2])
        self.assertEqual(exp_idxs, idxs)

    def test_process_frame_disappear_obj(self):
        objs1 = [obj1, obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj5]
        objs3 = [obj2, obj3, obj4, obj5]
        exp_idxs = [[0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        idxs = self._process_frames([objs1, objs2, objs3])
        self.assertEqual(exp_idxs, idxs)

    def test_process_frame_hidden_obj(self):
        objs1 = [obj1, obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj5]
        objs3 = [obj2, obj3, obj4, obj5, obj1]
        exp_idxs = [[0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4, 0]]
        idxs = self._process_frames([objs1, objs2, objs3])
        self.assertEqual(exp_idxs, idxs)

    def _process_frames(self, frames):
        idxs = []
        for i in range(len(frames)):
            ids = self.tracker.process_frame(frames[i])
            idxs.append(ids)

        return idxs
