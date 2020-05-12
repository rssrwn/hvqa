from pathlib import Path

from hvqa.util.interfaces import Component
from hvqa.util.asp_runner import ASPRunner


class ASPEventDetector(Component):
    def __init__(self, spec, al_model=True):
        self.spec = spec
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

    def eval(self, eval_data, tracker):
        """
        Evaluate the event detection component

        :param eval_data: Evaluation dataset (QADataset)
        :param tracker: ObjectTracker with run_ method for adding ids to objects
        """

        print("Evaluating hardcoded ASP event detector...")

        data = [eval_data[idx] for idx in range(len(eval_data))]
        videos, _ = tuple(zip(*data))

        action_set = set(self.spec.actions)

        total = 0
        total_correct = 0
        videos_correct = 0
        num_videos = 0

        for video_idx, video in enumerate(videos):
            tracker.run_(video)
            pred_events = self._detect_events(video.frames)
            pred_actions = []
            for events in pred_events:
                if len(events) == 1:
                    pred_actions.append(events[0][1])
                elif len(events) == 0:
                    print("No actions detected")
                    pred_actions.append("nothing")
                else:
                    print(f"Multiple actions detected: {events}")
                    pred_actions.append(None)

            act_actions = []
            for events in video.eval_events:
                actions = [event for event in events if event in action_set]
                if len(actions) == 1:
                    act_actions.append(actions[0])
                elif len(actions) == 0:
                    act_actions.append("nothing")
                else:
                    print(f"Multiple actions from dataset: {actions}")
                    act_actions.append(None)

            video_correct = 0

            for frame_idx, pred_action in enumerate(pred_actions):
                action = act_actions[frame_idx]
                if self.spec.from_internal("action", pred_action) == action:
                    video_correct += 1
                    total_correct += 1
                else:
                    print(f"Video {video_idx}, frame {frame_idx}: predicted {pred_action} -- actual {action}")

                total += 1
            num_videos += 1
            videos_correct += 1 if video_correct == 31 else 0

        event_acc = (total_correct / total) * 100
        video_acc = (videos_correct / num_videos) * 100
        print(f"Event accuracy: {event_acc:.2f}%")
        print(f"Video accuracy: {video_acc:.2f}%")
