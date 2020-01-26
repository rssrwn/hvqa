import os
import json
import argparse
import shutil
from PIL import Image
from pathlib import Path

from hvqa.dataset.video import Video
from hvqa.dataset.draw import Drawer
from hvqa.dataset.definitions import *


def write_json(out_dir):
    print("Writing json to file...")

    num_videos = 0
    for video_num in range(NUM_VIDEOS):
        # Create video
        video_builder = Video()
        video_builder.random_video()
        video = video_builder.to_dict()

        # Write text to file
        text = json.dumps(video)
        video_dir = Path(f"./{out_dir}/{video_num}")
        if not video_dir.exists():
            os.mkdir(f"./{out_dir}/{video_num}")

        file = open(f"./{out_dir}/{video_num}/video.json", "w")
        file.write(text)
        file.close()

        num_videos += 1

    print(f"Successfully written {num_videos} json files")


def create_videos(out_dir, verbose):
    basepath = Path(out_dir)
    video_dirs = basepath.iterdir()

    print("Creating frames from json...")

    num_frames_total = 0
    num_videos = 0
    for video_dir in video_dirs:
        json_file = video_dir / "video.json"
        if json_file.exists():
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            num_frames_video = 0
            frames = video_dict["frames"]
            for i, frame in enumerate(frames):
                img = create_frame(frame)
                img.save(f"{video_dir}/frame_{i}.png")
                num_frames_total += 1
                num_frames_video += 1

            if verbose and num_videos % 100 == 0:
                print(f"Processing video {num_videos}")

        else:
            print(f"No 'video.json' file found for {video_dir}/")

        num_videos += 1

    print(f"Successfully created {num_frames_total} frames")


def create_frame(frame):
    np_img = Drawer.draw_frame(frame)
    img = Image.fromarray(np_img, "RGB")
    return img


def delete_directory(name):
    response = input(f"About to delete {name} directory. Are you sure you want to continue? [y/n]")
    if response != "y":
        print("Exiting...")
        exit()

    directory = Path(f"./{name}")
    if directory.exists():
        try:
            shutil.rmtree(name)
        except OSError as e:
            print("Error while deleting directory: %s - %s." % (e.filename, e.strerror))


def main(out_dir, json_only, frames_only, verbose):
    if not frames_only:
        delete_directory(out_dir)
        os.mkdir(f"./{out_dir}")
        write_json(out_dir)

    if not json_only:
        response = input(f"About to create frames. This could overwrite old frames. "
                         f"Are you sure you want to continue? [y/n]")
        if response != "y":
            print("Exiting...")
            exit()

        create_videos(out_dir, verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for building dataset")
    parser.add_argument("-j", "--json", action="store_true", default=False)
    parser.add_argument("-f", "--frames", action="store_true", default=False)
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    main(args.out_dir, args.json, args.frames, args.verbose)
