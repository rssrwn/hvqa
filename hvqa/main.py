import argparse

import hvqa.util.func as util
from hvqa.videos import VideoDataset
from hvqa.detection.models import DetectionBackbone, DetectionModel
from hvqa.detection.detector import NeuralDetector
from hvqa.properties.models import PropertyExtractionModel
from hvqa.properties.prop_extractor import NeuralPropExtractor
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.coordination.coordinator import Coordinator


def simulate(dataset, coordinator):
    video, video_dict = dataset[4]
    coordinator.analyse_video(video)


def build_detector(detector_model_dir):
    backbone = DetectionBackbone()
    detector_model = util.load_model(DetectionModel, detector_model_dir, backbone)
    detector_model.eval()
    detector = NeuralDetector(detector_model)
    return detector


def build_prop_extractor(prop_model_dir):
    prop_model = util.load_model(PropertyExtractionModel, prop_model_dir)
    prop_model.eval()
    prop_extractor = NeuralPropExtractor(prop_model)
    return prop_extractor


def build_tracker():
    tracker = ObjTracker()
    return tracker


def main(data_dir, detector_model_dir, prop_model_dir):
    # Fetch data
    data = VideoDataset(data_dir)

    # Build pipeline components
    detector = build_detector(detector_model_dir)
    prop_extractor = build_prop_extractor(prop_model_dir)
    tracker = build_tracker()

    # Build coordinator
    coord = Coordinator(detector, prop_extractor, tracker)

    # Run pipeline
    simulate(data, coord)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for running VideoQA pipeline on a video")
    parser.add_argument("data_dir", type=str)
    parser.add_argument("detector_model_dir", type=str)
    parser.add_argument("prop_model_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.detector_model_dir, args.prop_model_dir)
