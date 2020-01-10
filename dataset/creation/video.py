from dataset.creation.frame import Frame


NUM_FRAMES = 10


class Video:

    def __init__(self):
        self.frames = []
        self.events = []

    def random_video(self):
        initial = Frame()
        initial.random_frame()
        self.frames.append(initial)

        curr = initial
        for frame in range(1, NUM_FRAMES):
            curr, events = curr.move()
            self.frames.append(curr)
            self.events.append(events)

        # TODO question and answer generation

    def to_dict(self):
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "events": self.events
        }
