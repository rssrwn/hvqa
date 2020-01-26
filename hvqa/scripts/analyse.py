import json
import argparse
from pathlib import Path


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


def get_video_dicts(data_dir):
    directory = Path(data_dir)

    dicts = []
    num_dicts = 0
    for video_dir in directory.iterdir():
        json_file = video_dir / "video.json"
        if json_file.exists():
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            dicts.append(video_dict)
            num_dicts += 1

    print(f"Successfully extracted {num_dicts} video dictionaries from json files")
    return dicts


def main(data_dir, events, colours, rotations):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for analysing built dataset")
    parser.add_argument("-e", "--events", action="store_true", default=False)
    parser.add_argument("-c", "--colours", action="store_true", default=False)
    parser.add_argument("-r", "--rotations", action="store_true", default=False)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.events, args.colours, args.rotations)
