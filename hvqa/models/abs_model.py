# Abstract QA Model Class File

import time


class _AbsModel:
    """
    Abstract QA Model Class
    Most of the models will run the same pipeline which is contained in this class
    Models need to set the components within their __init__ functions
    Models can also override methods if required
    """

    def __init__(self):
        self.obj_detector = self._setup_obj_detector()
        self.prop_classifier = self._setup_prop_classifier()
        self.tracker = self._setup_tracker()
        self.relation_classifier = self._setup_relation_classifier()
        self.event_detector = self._setup_event_detector()
        self.qa_system = self._setup_qa_system()

        self._timings = {
            "Detector": 0,
            "Properties": 0,
            "Tracker": 0,
            "Relations": 0,
            "Events": 0,
            "QA": 0
        }

    def _setup_obj_detector(self):
        raise NotImplementedError

    def _setup_prop_classifier(self):
        raise NotImplementedError

    def _setup_tracker(self):
        raise NotImplementedError

    def _setup_relation_classifier(self):
        raise NotImplementedError

    def _setup_event_detector(self):
        raise NotImplementedError

    def _setup_qa_system(self):
        raise NotImplementedError

    def run(self, frames, questions, q_types):
        """
        Generate answers to given questions

        :param frames: List of PIL Images
        :param questions: Questions about video: [str]
        :param q_types: Type of each question: [int]
        :return: Answers: [str]
        """

        video = self.process_frames(frames)
        qa_args = (video, questions, q_types)
        answers = self._time_func(self._answer_questions, qa_args, "QA")
        return answers

    def process_frames(self, frames):
        # Batch all frames in a video
        video = self._time_func(self._extract_objs, (frames,), "Detector")

        self.tracker.reset()

        # Batch all objects in each frame
        for frame in video.frames:
            objs = frame.objs
            self._time_func(self._extract_props_, (objs,), "Properties")
            ids = self._time_func(self.tracker.process_frame, (objs,), "Tracker")
            self._add_ids(objs, ids)
            self._time_func(self._detect_relations_, (frame,), "Relations")

        self._time_func(self._detect_events_, (video,), "Events")

        return video

    @staticmethod
    def _add_ids(objs, ids):
        for idx, obj in enumerate(objs):
            obj.id = ids[idx]

    def _extract_objs(self, frames):
        """
        Builds structured knowledge from video frames
        Extracts bboxs and class for each object in each frame

        :param frames: List of PIL frames
        :return: Video object for video
        """

        video = self.obj_detector.detect_objs(frames)
        return video

    def _extract_props_(self, objs):
        """
        Extract properties from each object and add them to the object
        Note: Modifies objects in-place with new properties

        :param objs: List of Objs
        """

        obj_imgs = [obj.img for obj in objs]
        props = self.prop_classifier.extract_props(obj_imgs)
        for idx, (colour, rot, cls) in enumerate(props):
            obj = objs[idx]
            obj.colour = colour
            obj.rot = rot

            # We use class from detector (if this line is commented)
            # obj.cls = cls

    def _detect_relations_(self, frame):
        """
        Extract relations between objects in a frame
        Note: Modifies frame in-place to add relations

        :param frame: Frame obj
        """

        objs = frame.objs
        rels = self.relation_classifier.detect_relations(objs)
        for idx1, idx2, rel in rels:
            frame.set_relation(idx1, idx2, rel)

    def _detect_events_(self, video):
        """
        Extract events from the video
        Note: Modifies video in-place to add event info

        :param video: Video obj
        """

        frames = video.frames
        events = self.event_detector.detect_events(frames)
        for frame_idx, frame_events in enumerate(events):
            for obj_id, event in frame_events:
                video.add_event(event, obj_id, frame_idx)

    def _answer_questions(self, video, questions, q_types):
        """
        Generate an answer for each question

        :param video: Video obj
        :param questions: Questions: [str]
        :param q_types: Question types: [int]
        :return: Answers: [str]
        """

        answers = self.qa_system.answer(video, questions, q_types)
        return answers

    def _time_func(self, func, args, timings_key):
        start = time.time()
        result = func(*args)
        total = time.time() - start
        self._timings[timings_key] += total
        return result

    def print_timings(self):
        timings = list(self._timings.items())
        sorted(timings, key=lambda pair: pair[1], reverse=True)
        total = sum(list(map(lambda pair: pair[1], timings)))

        print("Timings:")
        print(f"{'Component':<15}{'Time':<15}Share")
        for component, t in timings:
            print(f"{component:<15}{t:<15.2f}{t / total:.1%}")
        print()
