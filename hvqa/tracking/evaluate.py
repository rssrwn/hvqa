import argparse
import numpy as np

from hvqa.tracking.obj_tracker import ObjTracker


def eval(dataset):
    img1, _ = dataset[1000]
    imgs = [dataset[i][0] for i in range(10)]

    tracker = ObjTracker()

    fts = [tracker._extract_features(img) for img in imgs]
    fts = np.array(fts)

    ft1 = tracker._extract_features(img1)

    tracker._match_features(ft1[None, :], fts)


def main(data_dir):
    dataset = VideoDataset(data_dir)
    eval(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating the performance of the object tracking module")
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir)
