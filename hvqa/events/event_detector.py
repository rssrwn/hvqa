from pathlib import Path
import clingo


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

        # Add files
        ctl = clingo.Control()
        ctl.load(str(self.al_model))
        ctl.load(str(self.detector))
        ctl.load(str(self._video_info))

        # Configure the solver
        config = ctl.configuration
        config.solve.models = 0
        config.solve.opt_mode = "optN"

        ctl.ground([("base", [])])

        # Solve AL model with video info
        models = []
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if model.optimality_proven:
                    models.append(model.symbols(shown=True))

        assert len(models) != 0, "ASP event detection program is unsatisfiable"
        assert len(models) == 1, "ASP event detection program must contain only a single answer set"

        model = models[0]

        # Parse event info from ASP result
        events = [[]] * (len(frames) - 1)
        for sym in model:
            if sym.name == "occurs":
                event, frame = sym.arguments
                frame = frame.number
                event_name = event.name
                obj_id = event.arguments[0].number
                events[frame] = [(obj_id, event_name)]

        # Cleanup temp file
        # self._video_info.unlink()

        return events
