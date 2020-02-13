import numpy as np
from skimage.color import rgb2gray
from skimage.feature import ORB, hog, match_descriptors
from skimage import io


class ObjTracker:
    """
    Class used for assigning ids to each object in a frame, and tracking objects through frames.
    Ids are assigned initially and then the class will attempt to track objects in successive frames
    and assign their respective ids
    """

    def __init__(self):
        self._ids = None
        self._features = None
        # self._detector = cv2.ORB()
        self._detector = ORB(n_keypoints=10)

    def process_frame(self, objs):
        """
        Process a frame (list of objects)

        :param objs: List of objects which each take the form of a dictionary:
        obj: {
          image: PIL image of object,
        }
        :return: Ids for each for object
        """

        if self._ids is None:
            self._initial_frame(objs)
        else:
            self._process_frame(objs)

    def _initial_frame(self, objs):
        self._ids = list(range(len(objs)))
        self._features = [self._extract_features(obj["image"]) for obj in objs]

    def _process_frame(self, objs):
        pass

    def _extract_features(self, obj_img):
        img = np.asarray(obj_img, dtype=np.float32)
        img = rgb2gray(img)
        io.imshow(img)
        io.show()
        self._detector.detect_and_extract(img)
        descs = self._detector.descriptors
        return descs
