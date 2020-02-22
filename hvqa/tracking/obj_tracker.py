import math
from collections import deque

# import numpy as np
# from skimage.color import rgb2grey
# from skimage.feature import hog
# import torchvision.transforms as T


class ObjTracker:
    """
    Class used for assigning ids to each object in a frame, and tracking objects through frames.
    Ids are assigned initially and then the class will attempt to track objects in successive frames
    and assign their respective ids
    """

    def __init__(self):
        self._next_id = None
        self._objs = None
        self.frame_num = None
        self._max_hidden_frames = 5
        self._hidden_objects = deque()
        self._timeouts = deque()

    def process_frame(self, objs):
        """
        Process a frame (list of objects)
        Returns indices into the list of objects previously passed in.
        These indices correspond to the tracker's best guess of the object index that was in the previous frame

        :param objs: List of objects which each take the form of a dictionary:
        obj: {
          image: PIL image of object,
          class: type of object,
          position: (x1, y1, x2, y2)
        }
        :return: List indices (len = len(objs))
        """

        if self._objs is None:
            return self._initial_frame(objs)
        else:
            return self._process_next_frame(objs)

    def reset_ids(self):
        self._objs = None

    def _initial_frame(self, objs):
        self._objs = objs
        self.frame_num = 0
        ids = list(range(len(objs)))
        self._next_id = len(objs)
        return ids

    def _process_next_frame(self, objs):
        # Best matching index into <self._objs> for each curr obj
        idxs = [self._find_best_match(curr_obj, enumerate(self._objs)) for curr_obj in objs]

        # For each old obj we find all the curr objs which have matched with it
        matches = [[] for _ in self._objs]
        for curr_obj_idx, idx in enumerate(idxs):
            if curr_obj_idx is not None:
                matches[idx].append(curr_obj_idx)

        # Choose the single best curr obj for each old obj
        # Set every other new object to None in idxs list
        disappeared = []
        for idx, curr_obj_idxs in enumerate(matches):
            obj = self._objs[idx]
            curr_objs = [(idx, objs[idx]) for idx in curr_obj_idxs]
            match_idx = self._find_best_match(obj, curr_objs)

            if not curr_obj_idxs:
                disappeared.append((idx, self._objs[idx]))

            for curr_obj_idx in curr_obj_idxs:
                if curr_obj_idx != match_idx:
                    idxs[curr_obj_idx] = None

        # Check each new obj against hidden objs, and assign it the hidden objects id (if matched)
        # Otherwise assign a new id to each new obj
        new_objs = [(idx, objs[idx]) for idx, obj_idx in enumerate(idxs) if obj_idx is None]
        hidden_idxs = [self._find_best_match(new_obj, self._hidden_objects) for idx, new_obj in new_objs]
        for i, (idx, _) in enumerate(new_objs):
            if hidden_idxs[i] is not None:
                idxs[idx] = hidden_idxs[i]
            else:
                idxs[idx] = self._next_id
                self._next_id += 1

        self._update_hidden(disappeared)
        self._objs = objs
        self.frame_num += 1

        return idxs

    def _find_best_match(self, obj, objs):
        """
        Find the object in <objs> which best matches <obj>

        :param obj: Object dict
        :param objs: List of pairs [(idx, obj dict)]
        :return: Idx of best matching object
        """

        match_objs = []
        for idx, obj_ in objs:
            if obj["class"] == obj_["class"] and self.close_obj(obj, obj_):
                match_objs.append((idx, obj_))

        if len(match_objs) == 0:
            idx = None
        elif len(match_objs) == 1:
            idx = match_objs[0][0]
        else:
            dists = [(idx, self.dist(obj, obj_)) for idx, obj_ in match_objs]
            dists = sorted(dists, key=lambda idx_dist: idx_dist[1])
            idx = dists[0][0]

        return idx

    def _update_hidden(self, disappeared):
        frame_to_remove = self.frame_num - self._max_hidden_frames
        while len(self._timeouts) > 0:
            if self._timeouts[0] <= frame_to_remove:
                self._hidden_objects.popleft()
                self._timeouts.popleft()
            else:
                break

        self._hidden_objects.extend(disappeared)
        self._timeouts.extend([self.frame_num] * len(disappeared))

    @staticmethod
    def close_obj(obj1, obj2):
        x1, y1, _, _ = obj1["position"]
        x2, y2, _, _ = obj2["position"]
        close = abs(x1 - x2) <= 10 and abs(y1 - y2) <= 10
        return close

    @staticmethod
    def dist(obj1, obj2):
        x1, y1, _, _ = obj1["position"]
        x2, y2, _, _ = obj2["position"]
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    # def _extract_features(self, obj_img):
    #     img = self._resize(obj_img)
    #     img = np.asarray(img, dtype=np.int32)
    #     img = rgb2grey(img)
    #     features = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
    #     return features
