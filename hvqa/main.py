import argparse

import hvqa.util.util as util
from hvqa.util.definitions import detector_transforms, prop_transforms
from hvqa.videos import VideoDataset
from hvqa.detection.models import DetectionBackbone, DetectionModel
from hvqa.objprops.models import PropertyExtractionModel
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.coordinator import Coordinator


def simulate(dataset, coordinator):
    video, video_dict = dataset[4]
    coordinator.analyse_video(video)


def main(data_dir, detector_model_dir, prop_model_dir):
    data = VideoDataset(data_dir)
    backbone = DetectionBackbone()
    detector = util.load_model(DetectionModel, detector_model_dir, backbone)
    detector.eval()
    prop_model = util.load_model(PropertyExtractionModel, prop_model_dir)
    prop_model.eval()
    tracker = ObjTracker()
    coord = Coordinator(detector, detector_transforms, prop_model, prop_transforms, tracker)
    simulate(data, coord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for running VideoQA pipeline on a video")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("detector_model_dir", type=str)
    parser.add_argument("prop_model_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.detector_model_dir, args.prop_model_dir)
