import os
import json
import argparse
from PIL import Image
from pathlib import Path

from video import Video
from draw import Drawer
from definitions import *


def write_json(out_dir):
    print("Writing json to file...")
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

    print("Written json to file successfully")


def create_videos(out_dir):
    basepath = Path(out_dir)
    video_dirs = basepath.iterdir()

    print("Creating frames from json...")

    num_frames_total = 0
    for video_dir in video_dirs:
        json_file = basepath / video_dir / "video.json"
        if json_file.exists():
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            num_frames_video = 0
            for i in range(NUM_FRAMES):
                frame = video_dict["frames"][i]
                img = create_frame(frame)
                img.save(f"{out_dir}/{video_dir}/frame_{i}.png")
                num_frames_total += 1
                num_frames_video += 1

            if num_frames_video != NUM_FRAMES:
                print(f"Only {num_frames_video} created for video directory {video_dir}")

        else:
            print(f"No 'video.json' file found for {basepath}/{video_dir}/")

    print(f"Successfully created {num_frames_total} frames")


def create_frame(frame):
    np_img = Drawer.draw_frame(frame)
    img = Image.fromarray(np_img, "RGB")
    return img


def main(out_dir, json_only, frames_only):
    if json_only and frames_only:
        print("There is nothing to do... Check your arguments.")

    if json_only:
        response = input(f"About to create json files. "
                         f"This will overwrite existing json files in {out_dir}. "
                         f"Are you sure? [y]")
        if response != "y":
            print("Exiting...")

    if frames_only:
        response = input(f"About to create the frames from json files. "
                         f"This will overwrite existing frames in {out_dir}. "
                         f"Are you sure? [y]")
        if response != "y":
            print("Exiting...")

    if not json_only and not frames_only:
        response = input(f"About to create json files and frames. "
                         f"This will overwrite existing files in {out_dir}. "
                         f"Are you sure? [y]")
        if response != "y":
            print("Exiting...")

    if not frames_only:
        write_json(out_dir)

    if not json_only:
        create_videos(out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for building dataset")
    parser.add_argument("-j", "--json", action="store_true", default=False)
    parser.add_argument("-v", "--video", action="store_true", default=False)
    parser.add_argument("out_dir", type=str)
    args = parser.parse_args()
    main(args.out_dir, args.json, args.video)
