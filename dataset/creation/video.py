from dataset.creation.frame import Frame


NUM_FRAMES = 5


class Video:

    def __init__(self):
        self.frames = []

    def random_video(self):
        initial = Frame()
        initial.random_frame()
        self.frames.append(initial)

        curr = initial
        for frame in range(NUM_FRAMES):
            curr = curr.move_octopus()
            self.frames.append(curr)
