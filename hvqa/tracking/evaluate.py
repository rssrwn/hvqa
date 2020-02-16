import argparse

import hvqa.util as util
from hvqa.tracking.obj_tracker import ObjTracker
from hvqa.videos import VideoDataset


def generate_tracker_input(frame_img, frame_dict):
    imgs = []
    for obj in frame_dict["objects"]:
        img = util.collect_obj(frame_img, obj["position"])
        frame = {
            "image": img,
            "class": obj["class"],
            "position": obj["position"]
        }
        imgs.append(frame)

    return imgs


def close_obj(obj1, obj2):
    x1_1, _, x2_1, _ = obj1["position"]
    x1_2, _, x2_2, _ = obj2["position"]
    close = abs(x1_1 - x1_2) <= 10 and abs(x2_1 - x2_2) <= 10
    return close


def valid_rotation(obj1, obj2):
    obj1_rot = obj1["rotation"]
    obj2_rot = obj2["rotation"]
    rot1 = (obj1_rot + 1) % 4
    rot2 = (obj1_rot - 1) % 4
    valid = obj2_rot == rot1 or obj2_rot == rot2 or obj2_rot == obj1_rot
    return valid


def match_obj(obj1, obj2):
    obj1_type = obj1["class"]
    obj2_type = obj2["class"]

    if not obj1_type == obj2_type:
        return False

    if obj1_type != "octopus" and obj1["colour"] != obj2["colour"]:
        return False

    if not valid_rotation(obj1, obj2):
        return False

    if not close_obj(obj1, obj2):
        return False

    return True


def eval_tracking(objs, matched_objs):
    num_matched = 0
    for idx, obj in enumerate(matched_objs):
        orig_obj = objs[idx]
        if match_obj(orig_obj, obj):
            num_matched += 1
        else:
            print("Incorrect match")
            print(objs)
            print(matched_objs)

    return num_matched, len(matched_objs)


def evaluate(dataset):
    tracker = ObjTracker()

    print("Evaluating object tracker performance...")
    print(f"{'Video':<8}{'Number correct':<17}{'Total objects':<15}{'Accuracy':<10}")

    # Apply each video to the tracker
    for video_idx in range(len(dataset)):
        imgs, video_dict = dataset[video_idx]
        frames = video_dict["frames"]

        # Setup initial object list ordered by id from tracker
        frame = generate_tracker_input(imgs[0], frames[0])
        ids = tracker.process_frame(frame)
        objs = [frames[0]["objects"][id_] for id_ in ids]

        total_objs = 0
        num_correct = 0

        # Apply each following frame in the video to the tracker
        for frame_idx, img in enumerate(imgs[1:]):
            frame = generate_tracker_input(img, frames[frame_idx])
            tracked_ids = tracker.process_frame(frame)

            # Analyse the assignment
            frame_objects = frames[frame_idx]["objects"]
            matched_objs = [objs[id_] for id_ in tracked_ids]
            num_matched, num_objs = eval_tracking(frame_objects, matched_objs)
            num_correct += num_matched
            total_objs += num_objs

            objs = frame_objects

        acc = num_correct / total_objs

        print(f"{video_idx:<8}{num_correct:<17}{total_objs:<15}{acc:<10.4f}")

        tracker.reset_ids()


def main(data_dir):
    dataset = VideoDataset(data_dir)
    evaluate(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for evaluating the performance of the object tracking module")
    parser.add_argument("data_dir", type=str)
    args = parser.parse_args()
    main(args.data_dir)
