import math
from collections import deque


class ObjTracker:
    """
    Class used for assigning ids to each object in a frame, and tracking objects through frames.
    Ids are assigned initially and then the class will attempt to track objects in successive frames
    and assign their respective ids
    """

    def __init__(self, err_corr=False):
        self.err_corr = err_corr
        self.reset()

    def reset(self):
        self._next_id = None
        self._objs = None
        self.frame_num = None
        self._max_hidden_frames = 5
        self._max_movement = 20
        self._hidden_objects = deque()
        self._timeouts = deque()

    def process_frame_(self, objs):
        """
        Process a frame (list of objects)
        Returns object ids which correspond to the tracker's best guess of the object in the frame
        Note: Updates objs in-place

        :param objs: List of Obj objects
        :return: List indices (len = len(objs))
        """

        if self._objs is None:
            ids = self._initial_frame(objs)
        else:
            if self.err_corr:
                ids = self._process_next_frame_err_corr(objs)
            else:
                ids = self._process_next_frame(objs)

        # Assign ids
        for idx, obj in enumerate(objs):
            obj.id = ids[idx]

    def _initial_frame(self, objs):
        self._objs = objs
        self.frame_num = 0
        ids = list(range(len(objs)))
        self._ids = ids
        self._next_id = len(objs)
        return ids

    def _process_next_frame_err_corr(self, objs):
        ids = []
        for obj in objs:
            matches = self._find_matches(obj, self._objs)

            # Assign a new id if the object is new (and can't be found in hidden objects)
            if len(matches) == 0:
                hidden_matches = self._find_matches(obj, self._hidden_objects)
                if len(hidden_matches) == 0:
                    id_ = self._next_id
                    self._next_id += 1
                else:
                    id_ = hidden_matches[0].id

            elif len(matches) == 1:
                id_ = matches[0].id

            # Assign a single id if there are multiple matches
            else:
                id_ = matches[0].id
                if len(set([obj_.id for obj_ in matches])) > 1:
                    print("WARNING: Multiple ids from previous frame match with current object")

            ids.append(id_)

        # Find disappeared objects
        old_ids = set([obj_.id for obj_ in self._objs])
        disappeared_ids = old_ids - set(ids)

        disappeared = []
        for obj in self._objs:
            if obj.id in disappeared_ids:
                disappeared.append(obj)

        self._update_hidden(disappeared)

        return ids

    def _process_next_frame(self, objs):
        # Best matching index into <self._objs> for each curr obj
        idxs = [self._find_best_match(curr_obj, enumerate(self._objs)) for curr_obj in objs]

        # For each old obj we find all the curr objs which have matched with it
        matches = [[] for _ in self._objs]
        for curr_obj_idx, idx in enumerate(idxs):
            if idx is not None:
                matches[idx].append(curr_obj_idx)

        # Choose the single best curr obj for each old obj
        # Set every other new object to None in idxs list
        disappeared = []
        for idx, curr_obj_idxs in enumerate(matches):
            obj = self._objs[idx]
            curr_objs = [(idx, objs[idx]) for idx in curr_obj_idxs]
            match_idx = self._find_best_match(obj, curr_objs)

            if not curr_obj_idxs:
                disappeared.append((self._ids[idx], self._objs[idx]))

            for curr_obj_idx in curr_obj_idxs:
                if curr_obj_idx != match_idx:
                    idxs[curr_obj_idx] = None

        ids = []
        for curr_obj_idx, idx in enumerate(idxs):
            # Object did not appear in previous frame
            if idx is None:
                new_obj = objs[curr_obj_idx]
                hidden_id = self._find_best_match(new_obj, self._hidden_objects)

                # If no hidden objects match assign a new id
                # Otherwise assign hidden id
                if hidden_id is None:
                    ids.append(self._next_id)
                    self._next_id += 1
                else:
                    ids.append(hidden_id)

            # Object did appear in previous frame -> assign same id
            else:
                id_ = self._ids[idx]
                ids.append(id_)

        self._update_hidden(disappeared)
        self._objs = objs
        self._ids = ids
        self.frame_num += 1

        return ids

    def _find_best_match(self, obj, objs):
        """
        Find the object in <objs> which best matches <obj>
        :param obj: Obj
        :param objs: List of pairs [(idx, Obj)]
        :return: Idx of best matching object
        """

        match_objs = []
        for idx, obj_ in objs:
            if obj.cls == obj_.cls and self.close_obj(obj, obj_):
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

    def _find_matches(self, obj, objs):
        """
        Find all objects in <objs> which match with <obj>
        Two objects match if they have the same class and are close (as defined in close_obj)

        :param obj: Obj
        :param objs: List of Objs
        :return: List of Objs
        """

        match_objs = []
        for obj_ in objs:
            if obj.cls == obj_.cls:
                if obj.is_static and self._close_pos(obj, obj_):
                    match_objs.append(obj_)
                elif (not obj.is_static) and self.close_obj(obj, obj_):
                    match_objs.append(obj_)

        match_objs = [(self.dist(obj, obj_), obj_) for obj_ in match_objs]
        match_objs = sorted(match_objs, key=lambda dist_obj: dist_obj[0])
        match_objs = [obj_ for dist, obj_ in match_objs]
        return match_objs

    def _find_close_matches(self, obj, objs):
        """
        Find all objects in <objs> which both match and are very close to <obj>

        :param obj: Obj
        :param objs: List of pairs [(id, Obj)]
        :return: List of ids of close, matched objects
        """

        match_objs = []
        for id_, obj_ in objs:
            if obj.cls == obj_.cls and self._close_pos(obj, obj_):
                match_objs.append(id_)

        return match_objs

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

    def close_obj(self, obj1, obj2):
        x1, y1, _, _ = obj1.pos
        x2, y2, _, _ = obj2.pos
        close = abs(x1 - x2) <= self._max_movement and abs(y1 - y2) <= self._max_movement
        return close

    def _close_pos(self, obj1, obj2):
        dist = self.dist(obj1, obj2)
        return dist <= 3

    @staticmethod
    def dist(obj1, obj2):
        x1, y1, _, _ = obj1.pos
        x2, y2, _, _ = obj2.pos
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
