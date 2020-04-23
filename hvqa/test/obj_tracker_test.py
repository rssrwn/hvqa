import unittest

from hvqa.util.environment import Obj
from hvqa.tracking.obj_tracker import ObjTracker


obj1 = Obj("octopus", (30, 30, 60, 60))
obj1_copy = Obj("octopus", (30, 30, 60, 60))
obj1_err = Obj("octopus", (31, 30, 61, 60))
obj1_moved = Obj("octopus", (45, 30, 75, 60))
obj1_err_moved = Obj("octopus", (46, 30, 76, 60))
obj2 = Obj("fish", (100, 30, 200, 200))
obj3 = Obj("octopus", (110, 40, 0, 0))
obj4 = Obj("fish", (30, 30, 60, 60))
obj5 = Obj("rock", (110, 40, 0, 0))


class ObjTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = ObjTracker()
        self.tracker_err_corr = ObjTracker(err_corr=True)

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
        objs = enumerate([obj1_moved, obj1_copy, obj2, obj3])
        exp_idx = 1
        idx = self.tracker._find_best_match(obj1, objs)
        self.assertEqual(exp_idx, idx, "Found wrong match for obj")

    def test_find_best_match_moved(self):
        objs = enumerate([obj2, obj3, obj1_moved])
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
        exp_ids = [0, 1, 2, 3, 4]
        self.tracker.process_frame_(objs)
        ids = [obj.id for obj in objs]
        self.assertEqual(exp_ids, ids)

    def test_process_frame_no_change(self):
        objs = [obj1, obj2, obj3, obj4, obj5]
        exp_ids = [0, 1, 2, 3, 4]
        for _ in range(2):
            self.tracker.process_frame_(objs)
            ids = [obj.id for obj in objs]
            self.assertEqual(exp_ids, ids)

    def test_process_frame_moved_obj(self):
        objs1 = [obj2, obj3, obj4, obj5, obj1]
        objs2 = [obj2, obj3, obj1_moved, obj4, obj5]
        exp_ids = [[0, 1, 2, 3, 4], [0, 1, 4, 2, 3]]
        ids = self._process_frames([objs1, objs2])
        self.assertEqual(exp_ids, ids)

    def test_process_frame_new_obj(self):
        objs1 = [obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj1, obj5]
        exp_ids = [[0, 1, 2, 3], [0, 1, 2, 4, 3]]
        ids = self._process_frames([objs1, objs2])
        self.assertEqual(exp_ids, ids)

    def test_process_frame_disappear_obj(self):
        objs1 = [obj1, obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj5]
        objs3 = [obj2, obj3, obj4, obj5]
        exp_ids = [[0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        ids = self._process_frames([objs1, objs2, objs3])
        self.assertEqual(exp_ids, ids)

    def test_process_frame_hidden_obj(self):
        objs1 = [obj1, obj2, obj3, obj4, obj5]
        objs2 = [obj2, obj3, obj4, obj5]
        objs3 = [obj2, obj3, obj4, obj5, obj1]
        exp_ids = [[0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4, 0]]
        ids = self._process_frames([objs1, objs2, objs3])
        self.assertEqual(exp_ids, ids)

    def test_err_corr_appear_disappear(self):
        objs1 = [obj1, obj2, obj5]
        objs2 = [obj1, obj1_err, obj2, obj5]
        objs3 = [obj1, obj2, obj5]
        exp_ids = [[0, 1, 2], [0, 0, 1, 2], [0, 1, 2]]
        ids = self._process_frames([objs1, objs2, objs3], err_corr=True)
        self.assertEqual(exp_ids, ids)

    def test_err_corr_move(self):
        objs1 = [obj1, obj2, obj5]
        objs2 = [obj1, obj1_err, obj2, obj5]
        objs3 = [obj1_moved, obj1_err_moved, obj2, obj5]
        objs4 = [obj1, obj2, obj5]
        exp_ids = [[0, 1, 2], [0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 2]]
        ids = self._process_frames([objs1, objs2, objs3, objs4], err_corr=True)
        self.assertEqual(exp_ids, ids)

    def _process_frames(self, frames, err_corr=False):
        idxs = []
        for i in range(len(frames)):
            frame = frames[i]
            if err_corr:
                self.tracker_err_corr.process_frame_(frame)
            else:
                self.tracker.process_frame_(frame)

            ids = [obj.id for obj in frame]
            idxs.append(ids)

        return idxs
