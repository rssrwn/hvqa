from hvqa.util.interfaces import Component


class _AbsEventDetector(Component):
    def __init__(self, spec):
        self.spec = spec

    def run_(self, video):
        frames = video.frames
        events = self._detect_events(frames)
        for frame_idx, frame_events in enumerate(events):
            for obj_id, event in frame_events:
                video.add_action(event, obj_id, frame_idx)

    @staticmethod
    def new(spec, **kwargs):
        raise NotImplementedError()

    def _detect_events(self, frames):
        """
        Detect events between frames
        Returns a list of length len(frames) - 1
        Each element, i, is a list of events which occurred between frame i and i+1

        :param frames: List of Frame objects
        :return: List of List of (id: int, event_name: str)
        """

        raise NotImplementedError()

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
