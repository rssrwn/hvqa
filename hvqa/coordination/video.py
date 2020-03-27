from hvqa.util.util import collect_obj
from hvqa.util.definitions import relations, events, VIDEO_LENGTH


class Obj:
    def __init__(self, cls, pos):
        self.cls = cls
        self.pos = pos
        self.rot = None
        self.colour = None
        self.id = None
        self.img = None

    def set_image(self, img):
        """
        Set the img param for this object by cropping the frame with object position

        :param img: Full frame
        """

        assert self.pos is not None, "Position must be set in order to extract object image"
        self.img = collect_obj(img, self.pos)


class Frame:
    def __init__(self, objs):
        self.objs = objs
        self.relations = []
        self._obj_id_map = {obj.id: obj for obj in objs}

    def set_relation(self, idx1, idx2, relation):
        assert relation in relations, f"Relation arg must be one of {relations}"
        self.relations.append((idx1, idx2, relation))


class Video:
    def __init__(self, frames):
        self.frames = frames
        self.events = [None] * (VIDEO_LENGTH - 1)

    def set_event(self, event, obj_id, start_idx):
        assert event in events, f"Event arg must be one of {events}"
        self.events[start_idx] = (event, obj_id)
