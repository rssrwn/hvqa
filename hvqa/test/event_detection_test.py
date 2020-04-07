import unittest

from hvqa.util.video_repr import Obj, Frame
from hvqa.events.event_detector import ASPEventDetector


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


class EventDetectionTest(unittest.TestCase):
    def setUp(self):
        self.detector = ASPEventDetector("hvqa/events")

    def test_detect_move(self):
        frames = [frame1, frame2]
        events = self.detector.detect_events(frames)
        exp_events = [[(0, "move")]]
        self.assertEqual(exp_events, events)

    def test_detect_rotate_right(self):
        frames = [frame2, frame3]
        events = self.detector.detect_events(frames)
        exp_events = [[(0, "rotate_right")]]
        self.assertEqual(exp_events, events)

    def test_detect_rotate_left(self):
        frames = [frame3, frame2]
        events = self.detector.detect_events(frames)
        exp_events = [[(0, "rotate_left")]]
        self.assertEqual(exp_events, events)

    def test_nothing_event(self):
        frames = [frame1, frame1]
        events = self.detector.detect_events(frames)
        exp_events = [[]]
        self.assertEqual(exp_events, events)

    def test_multiple_events(self):
        frames = [frame1, frame2, frame3]
        events = self.detector.detect_events(frames)
        exp_events = [[(0, "move")], [(0, "rotate_right")]]
        self.assertEqual(exp_events, events)
