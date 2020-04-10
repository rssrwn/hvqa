from clyngor import solve
from pathlib import Path


class _AbsEventDetector:
    def detect_events(self, frames):
        """
        Detect events between frames
        Returns a list of length len(frames) - 1
        Each element, i, is a list of events which occurred between frame i and i+1

        :param frames: List of Frame objects
        :return: List of List of (id: int, event_name: str)
        """

        raise NotImplementedError


class ASPEventDetector(_AbsEventDetector):
    def __init__(self, asp_dir):
        path = Path(asp_dir)
        self.al_model = path / "model.lp"
        self.detector = path / "events.lp"
        self._video_info = path / "_temp_video_info.lp"

        assert self.al_model.exists(), f"File {self.al_model} does not exist"
        assert self.detector.exists(), f"File {self.detector} does not exist"

    def detect_events(self, frames):
        # Create ASP file for video information
        asp_enc = ""
        for idx, frame in enumerate(frames):
            asp_enc += frame.gen_asp_encoding(idx) + "\n"

        f = open(self._video_info, "w")
        f.write(asp_enc)
        f.close()

        # Solve AL model with video info
        asp_events = solve([self.al_model, self.detector, self._video_info], use_clingo_module=True)
        asp_events = [asp_event for asp_event in asp_events]

        assert len(asp_events) != 0, "ASP event detection program is unsatisfiable"
        assert not len(asp_events) > 1, "ASP event detection program must contain only a single answer set"

        asp_events = asp_events[0]

        # Parse event info from ASP result
        events = [[]] * (len(frames) - 1)
        for pred, args in asp_events:
            if pred == "occurs":
                event, frame = args
                splits = event.split("(")
                event_name = splits[0]
                obj_id = int(splits[1][0:-1])  # Remove closing bracket
                events[frame] = [(obj_id, event_name)]

        # Cleanup temp file
        # self._video_info.unlink()

        return events
