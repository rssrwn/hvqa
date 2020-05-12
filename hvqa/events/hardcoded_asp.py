from pathlib import Path

from hvqa.events.abs_asp import _AbsEventDetector
from hvqa.util.asp_runner import ASPRunner


class ASPEventDetector(_AbsEventDetector):
    def __init__(self, spec, al_model=True):
        super(ASPEventDetector, self).__init__(spec)

        self.al_model = al_model

        path = Path("hvqa/events")
        self._video_info = path / "_temp_video_info.lp"

        if al_model:
            self.detector_file = path / "events.lp"
            self.al_model_file = path / "model.lp"
        else:
            self.detector_file = path / "occurs_events.lp"

        self.timeout = 5

        assert self.detector_file.exists(), f"File {self.detector_file} does not exist"
        if al_model:
            assert self.al_model_file.exists(), f"File {self.al_model_file} does not exist"

    def run_(self, video):
        frames = video.frames
        events = self._detect_events(frames)
        for frame_idx, frame_events in enumerate(events):
            for obj_id, event in frame_events:
                video.add_action(event, obj_id, frame_idx)

    @staticmethod
    def new(spec, **kwargs):
        al_model = kwargs["al_model"]
        events = ASPEventDetector(spec, al_model)
        return events

    def _detect_events(self, frames):
        """
        Detect events between frames
        Returns a list of length len(frames) - 1
        Each element, i, is a list of events which occurred between frame i and i+1

        :param frames: List of Frame objects
        :return: List of List of (id: int, event_name: str)
        """

        # Create ASP file for video information
        asp_enc = ""
        for idx, frame in enumerate(frames):
            asp_enc += frame.gen_asp_encoding(idx) + "\n"

        name = "Event detection"
        files = [self.detector_file]
        if self.al_model:
            files.append(self.al_model_file)

        models = ASPRunner.run(self._video_info, asp_enc, additional_files=files, timeout=self.timeout,
                               opt_mode="optN", opt_proven=self.al_model, prog_name=name)

        assert len(models) != 0, "ASP event detection program is unsatisfiable"

        if len(models) > 1:
            print("WARNING: Event detection ASP program contains multiple answer sets. Choosing one answer...")

        model = models[0]

        # Parse event info from ASP result
        events = [[]] * (len(frames) - 1)
        correct_objs = {}
        for sym in model:
            # Get actions from occurs predicate
            if sym.name == "occurs":
                event, frame = sym.arguments
                frame = frame.number
                event_name = event.name
                obj_id = event.arguments[0].number
                events[frame] = events[frame] + [(obj_id, event_name)]

            # Work out which objects are nn errors from err_id predicate
            elif sym.name == "try_obj":
                err_id, frame = sym.arguments
                err_id = err_id.number
                frame = frame.number
                frame_err_ids = correct_objs.get(frame)
                frame_err_ids = [] if frame_err_ids is None else frame_err_ids
                frame_err_ids.append(err_id)
                correct_objs[frame] = frame_err_ids

        # Set correct objects in each frame, error objects are removed
        for frame_num, try_ids in correct_objs.items():
            frame = frames[frame_num]
            frame.set_correct_objs(try_ids)

        return events
