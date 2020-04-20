from hvqa.util.func import collect_obj
from hvqa.util.definitions import RELATIONS, ACTIONS, VIDEO_LENGTH, CLASSES


class Obj:
    def __init__(self, cls, pos):
        assert cls in CLASSES, f"Class must be one of {CLASSES}"

        self.cls = cls
        self.pos = pos
        self.is_static = cls != "octopus"
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

    def gen_asp_encoding(self, frame_num, body=None):
        assert self.cls is not None, "Class must be set"
        assert self.pos is not None, "Position must be set"
        assert self.rot is not None, "Rotation must be set"
        assert self.colour is not None, "Colour must be set"
        assert self.id is not None, "Id must be set"

        self.pos = tuple(map(int, self.pos))

        body_str = "" if body is None else " :- " + body
        frame_num = str(frame_num)
        encoding = f"obs(class({self.cls}, {self.id}), {frame_num}){body_str}.\n" \
                   f"obs(position({str(self.pos)}, {self.id}), {frame_num}){body_str}.\n" \
                   f"obs(rotation({str(self.rot)}, {self.id}), {frame_num}){body_str}.\n" \
                   f"obs(colour({self.colour}, {self.id}), {frame_num}){body_str}.\n"

        return encoding


class Frame:
    def __init__(self, objs):
        self.objs = objs
        self.relations = []
        self._obj_id_map = {obj.id: obj for obj in objs}

    def set_relation(self, idx1, idx2, relation):
        assert relation in RELATIONS, f"Relation arg must be one of {RELATIONS}"

        id1 = self.objs[idx1].id
        id2 = self.objs[idx2].id
        self.relations.append((id1, id2, relation))

    def gen_asp_encoding(self, frame_num):
        err_idxs = self._find_duplicate_idxs()

        enc = ""

        err_id = 0
        idx_err_id_map = {}
        for id_, idxs in err_idxs.items():
            choice_str = "{ "
            for idx in idxs:
                choice_str += f"err_obj({err_id}), "
                idx_err_id_map[idx] = err_id
                err_id += 1

            choice_str = choice_str[:-2] + " }.\n"
            enc += choice_str

        for idx, obj in enumerate(self.objs):
            err_id = idx_err_id_map.get(idx)
            if err_id is not None:
                body_str = f"err_obj({err_id})"
                enc += obj.gen_asp_encoding(frame_num, body_str) + "\n"
            else:
                enc += obj.gen_asp_encoding(frame_num) + "\n"

        for id1, id2, rel in self.relations:
            enc += f"obs({rel}({str(id1)}, {str(id2)}), {str(frame_num)}).\n"

        return enc

    def _find_duplicate_idxs(self):
        ids = {}
        for idx, obj in enumerate(self.objs):
            idxs = ids.get(obj.id)
            idxs = [] if idxs is None else idxs
            idxs.append(idx)
            ids[obj.id] = idxs

        dup_ids = {id_: idxs for id_, idxs in ids.items() if len(idxs) > 1}
        return dup_ids


class Video:
    def __init__(self, frames):
        self.frames = frames
        self.events = [[]] * (VIDEO_LENGTH - 1)

    def add_event(self, event, obj_id, start_idx):
        assert event in ACTIONS, f"Event {event} is not one of {ACTIONS}"
        self.events[start_idx] = self.events[start_idx] + [(event, obj_id)]

    def gen_asp_encoding(self):
        enc = ""
        for frame_idx, frame in enumerate(self.frames):
            enc += frame.gen_asp_encoding(frame_idx)

        enc += "\n"

        for frame_idx, events in enumerate(self.events):
            for event, obj_id in events:
                enc += f"occurs({event}({obj_id}), {frame_idx}).\n"

        return enc
