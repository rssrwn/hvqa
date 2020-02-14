import numpy as np
import math
from skimage.color import rgb2grey
from skimage.feature import hog
import torchvision.transforms as T


class ObjTracker:
    """
    Class used for assigning ids to each object in a frame, and tracking objects through frames.
    Ids are assigned initially and then the class will attempt to track objects in successive frames
    and assign their respective ids
    """

    def __init__(self):
        self._ids = None
        self._objs = None
        self._resize = T.Resize((16, 16))

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
        ids = list(range(len(objs)))
        return ids

    def _process_next_frame(self, objs):
        idxs = []
        for new_obj in objs:
            match_objs = []
            for idx, obj in enumerate(self._objs):
                if new_obj["class"] == obj["class"] and self.close_obj(obj, new_obj):
                    match_objs.append((idx, obj))

            if len(match_objs) == 0:
                idx = None
            elif len(match_objs) == 1:
                idx = match_objs[0][0]
            else:
                dists = [(idx, self.dist(new_obj, obj)) for idx, obj in match_objs]
                dists = sorted(dists, key=lambda idx_dist: idx_dist[1])
                idx = dists[0][0]

            if idx is None:
                print("No matching objects")
            else:
                idxs.append(idx)

        self._objs = objs

        return idxs

    @staticmethod
    def close_obj(obj1, obj2):
        x1, y1, _, _ = obj1["position"]
        x2, y2, _, _ = obj2["position"]
        close = abs(x1 - x2) <= 20 and abs(y1 - y2) <= 20
        return close

    @staticmethod
    def dist(obj1, obj2):
        x1, y1, _, _ = obj1["position"]
        x2, y2, _, _ = obj2["position"]
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def _extract_features(self, obj_img):
        img = self._resize(obj_img)
        img = np.asarray(img, dtype=np.int32)
        img = rgb2grey(img)
        features = hog(img, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1))
        return features
