import json
import argparse
from pathlib import Path


def extract_event(event, from_colours, to_colours):
    if event[0:13] == "change colour":
        words = event.split(" ")
        from_colours.append(words[3])
        to_colours.append(words[5])
        event = "change colour"

    return event


def count_events(video_dicts):
    from_colours = []
    to_colours = []
    event_list = []
    num_frame_changes = 0
    for video in video_dicts:
        events = video["events"]
        num_frame_changes += len(events)
        flat_events = [event for subevents in events for event in subevents]
        events = [extract_event(event, from_colours, to_colours) for event in flat_events]
        event_list.extend(events)

    event_dict = {}
    for event in event_list:
        num_event = event_dict.get(event)
        if not num_event:
            num_event = 0

        event_dict[event] = num_event + 1

    print(f"\n{'Event name' :<20}{'Occurrences' :<15}Frequency")
    for event, num in event_dict.items():
        print(f"{event:<20}{num:<15}{(num / num_frame_changes) * 100:.3g}%")

    print(f"\nTotal number of frame changes: {num_frame_changes}\n")


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


def main(data_dir, events):
    video_dicts = get_video_dicts(data_dir)

    if events:
        print("Analysing event occurrences...")
        count_events(video_dicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for analysing built dataset")
    parser.add_argument("-e", "--events", action="store_true", default=False)
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir, args.events)
