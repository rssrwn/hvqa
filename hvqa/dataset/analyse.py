import argparse
import numpy as np

from hvqa.util import get_video_dicts, detector_transforms
from hvqa.detection.dataset import ClassificationDataset


def extract_event(event):
    if event[0:13] == "change colour":
        event = "change colour"

    return event


def increment_in_map(coll, key):
    curr_value = coll.get(key)
    if not curr_value:
        curr_value = 0

    coll[key] = curr_value + 1


def count_events(video_dicts):
    event_list = []
    num_frame_changes = 0
    for video in video_dicts:
        events = video["events"]
        num_frame_changes += len(events)
        flat_events = [event for subevents in events for event in subevents]
        events = [extract_event(event) for event in flat_events]
        event_list.extend(events)

    event_dict = {}
    for event in event_list:
        increment_in_map(event_dict, event)

    print(f"\n{'Event name' :<20}{'Occurrences' :<15}Frequency")
    for event, num in event_dict.items():
        print(f"{event:<20}{num:<15}{(num / num_frame_changes) * 100:.3g}%")

    print(f"\nTotal number of frame changes: {num_frame_changes}\n")


def count_colours(video_dicts):
    rock_colours = {}
    octo_colours = {}
    num_frames = 0
    for video in video_dicts:
        frames = video["frames"]
        for frame in frames:
            num_frames += 1
            objects = frame["objects"]
            for obj in objects:
                if obj["class"] == "rock":
                    increment_in_map(rock_colours, obj["colour"])
                elif obj["class"] == "octopus":
                    increment_in_map(octo_colours, obj["colour"])

    print(f"\n{'Rock colour' :<20}{'Occurrences' :<15}Frequency")
    for colour, num in rock_colours.items():
        print(f"{colour:<20}{num:<15}{(num / num_frames) * 100:.3g}%")

    print(f"\n{'Octopus colour' :<20}{'Occurrences' :<15}Frequency")
    for colour, num in octo_colours.items():
        print(f"{colour:<20}{num:<15}{(num / num_frames) * 100:.3g}%")

    print(f"\nTotal number of frames: {num_frames}\n")


def count_rotations(video_dicts):
    rotations = {}
    num_frames = 0
    for video in video_dicts:
        frames = video["frames"]
        for frame in frames:
            num_frames += 1
            objects = frame["objects"]
            for obj in objects:
                if obj["class"] == "octopus":
                    increment_in_map(rotations, obj["rotation"])

    print(f"\n{'Octopus rotations' :<20}{'Occurrences' :<15}Frequency")
    for rotation, num in rotations.items():
        print(f"{rotation:<20}{num:<15}{(num / num_frames) * 100:.3g}%")

    print(f"\nTotal number of frames: {num_frames}\n")


def analyse_pixels(dataset):
    img, target = dataset[0]
    num_channels = img.shape[0]
    pixels = [[] for _ in range(num_channels)]

    num_samples = 1000
    idxs = np.random.randint(0, len(dataset), (num_samples,))
    for idx in idxs:
        img, target = dataset[idx]
        img = img.numpy()
        for c_idx in range(num_channels):
            pixels[c_idx].append(list(img[c_idx, :, :].reshape(-1)))

    means = []
    stddevs = []
    for arr in pixels:
        means.append(np.array(arr).mean())
        stddevs.append(np.array(arr).std())

    print("Dataset statistics per channel:")
    print(f"Means:    {tuple(means)}")
    print(f"Std devs: {tuple(stddevs)}")


def main(data_dir, events, colours, rotations, pixels):
    video_dicts = get_video_dicts(data_dir)

    if events:
        print("Analysing event occurrences...")
        count_events(video_dicts)

    if colours:
        print("Analysing object colours...")
        count_colours(video_dicts)

    if rotations:
        print("Analysing octopus rotations...")
        count_rotations(video_dicts)

    if pixels:
        print("Analysing pixels...")
        dataset = ClassificationDataset(data_dir, transforms=detector_transforms)
        analyse_pixels(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for analysing built dataset")
    parser.add_argument("-e", "--events", action="store_true", default=False)
    parser.add_argument("-c", "--colours", action="store_true", default=False)
    parser.add_argument("-r", "--rotations", action="store_true", default=False)
    parser.add_argument("-p", "--pixels", action="store_true", default=False)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.events, args.colours, args.rotations, args.pixels)
