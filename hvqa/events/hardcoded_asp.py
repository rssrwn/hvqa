import time
import clingo
from pathlib import Path

from hvqa.util.interfaces import Component


class ASPEventDetector(Component):
    def __init__(self, al_model=True):
        self.al_model = al_model

        path = Path("hvqa/events")
        self._video_info = path / "_temp_video_info.lp"

        if al_model:
            self.detector_file = path / "events.lp"
            self.al_model_file = path / "model.lp"
        else:
            self.detector = path / "occurs_events.lp"

        self.timeout = 5

        assert self.al_model_file.exists(), f"File {self.al_model_file} does not exist"
        assert self.detector_file.exists(), f"File {self.detector_file} does not exist"

    def run_(self, video):
        frames = video.frames
        events = self._detect_events(frames)
        for frame_idx, frame_events in enumerate(events):
            for obj_id, event in frame_events:
                video.add_action(event, obj_id, frame_idx)

    @staticmethod
    def new(spec, **kwargs):
        al_model = kwargs["al_model"]
        events = ASPEventDetector(al_model)
        return events

    def train(self, train_data, eval_data, verbose=True):
        raise NotImplementedError()

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

        f = open(self._video_info, "w")
        f.write(asp_enc)
        f.close()

        # Add files
        ctl = clingo.Control(message_limit=0)
        ctl.load(str(self.detector_file))
        ctl.load(str(self._video_info))
        if self.al_model:
            ctl.load(str(self.al_model_file))

        # Configure the solver
        config = ctl.configuration
        config.solve.models = 0
        config.solve.opt_mode = "optN"

        ctl.ground([("base", [])])

        # Solve AL model with video info
        models = []
        start_time = time.time()
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                if self.al_model and model.optimality_proven:
                    models.append(model.symbols(shown=True))
                elif not self.al_model:
                    models.append(model.symbols(shown=True))

                if time.time() - start_time > self.timeout:
                    print("WARNING: Event detection program reached timeout")
                    handle.cancel()
                    break

        # Cleanup temp file
        self._video_info.unlink()

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