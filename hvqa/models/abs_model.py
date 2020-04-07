# Abstract QA Model Class File


class _AbsModel:
    """
    Abstract QA Model Class
    Most of the models will run the same pipeline which is contained in this class
    Models need to set the components within their __init__ functions
    Models can also override methods if required
    """

    def __init__(self):
        self.detector = None
        self.prop_classifier = None
        self.tracker = None
        self.relation_classifier = None
        self.event_classifier = None

    def run(self, frames, questions, q_types):
        """
        Generate answers to given questions

        :param frames: List of PIL Images
        :param questions: Questions about video: [str]
        :param q_types: Type of each question: [int]
        :return: Answers: [str]
        """

        video = self.process_frames(frames)
        pass

    def process_frames(self, frames):
        # Batch all frames in a video
        video = self._extract_objs(frames)

        self.tracker.reset()

        # Batch all objects in each frame
        for frame in video.frames:
            objs = frame.objs
            self._extract_props_(objs)
            ids = self.tracker.process_frame(objs)
            self._add_ids(objs, ids)
            self._detect_relations_(frame)

        self._detect_events_(video)

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

        video = self.detector.detect_objs(frames)
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
        events = self.event_classifier.detect_events(frames)
        for frame_idx, frame_events in enumerate(events):
            for obj_id, event in frame_events:
                video.add_event(event, obj_id, frame_idx)
