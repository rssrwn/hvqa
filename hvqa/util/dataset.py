import json
import time
from pathlib import Path
from more_itertools import grouper
from concurrent.futures import ThreadPoolExecutor

import hvqa.util.func as util
from hvqa.util.interfaces import QADataset
from hvqa.util.environment import Video, Frame, Obj


class VideoDataset(QADataset):
    """
    Dataset for storing and fetching videos
    """

    def __init__(self, spec, videos, answers, timing=0, hardcoded=False):
        super(VideoDataset, self).__init__()

        self.spec = spec
        self._detector_timing = timing
        self.hardcoded = hardcoded
        self.videos = videos
        self.answers = answers

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, item):
        """
        Get item of dataset

        :param item: Video number (as stored in directory)
        :return: Video obj, list of answers ([str])
        """

        video = self.videos[item]
        ans = self.answers[item]
        return video, ans

    def is_hardcoded(self):
        return self.hardcoded

    def detector_timing(self):
        return self._detector_timing

    @classmethod
    def from_data_dir(cls, spec, data_dir, detector, hardcoded=False, group_videos=8):
        data_dir = Path(data_dir)
        ids, videos, answers, timing = cls._find_videos(spec, data_dir, detector, hardcoded, group_videos)
        ids = sorted(enumerate(ids), key=lambda idx, id_: id_)
        videos = [videos[idx] for idx, _ in ids]
        answers = [answers[idx] for idx, _ in ids]
        dataset = VideoDataset(spec, videos, answers, timing=timing, hardcoded=hardcoded)
        return dataset

    @classmethod
    def _find_videos(cls, spec, data_dir, detector, hardcoded, group_videos):
        video_ids = []
        videos = []
        answers = []

        detector_timing = 0

        print(f"Searching directory {data_dir} for videos...")
        video_infos = cls._collect_videos(data_dir)
        print(f"Found data from {len(video_infos)} videos")
        print("Extracting objects...")

        # Group videos together for faster object detection
        for video_info in grouper(video_infos, group_videos):
            video_info = [info for info in video_info if info is not None]
            video_nums, video_dicts, frame_imgs = tuple(zip(*video_info))
            grouped_videos, timing = cls._construct_videos(spec, video_dicts, frame_imgs, detector, hardcoded)
            videos.extend(grouped_videos)
            video_ids.extend(video_nums)
            ans = [video_dict["answers"] for video_dict in video_dicts]
            answers.extend(ans)
            detector_timing += timing

        print("Completed object extraction.\n")

        return video_ids, videos, answers, detector_timing

    @classmethod
    def _construct_videos(cls, spec, video_dicts, imgs, detector, hardcoded):
        detector_timing = 0
        if hardcoded:
            videos_frames = []
            for idx, video_dict in enumerate(video_dicts):
                frame_imgs = imgs[idx]
                frame_dicts = video_dict["frames"]
                frames = []
                for frame_num, frame_dict in enumerate(frame_dicts):
                    frame_img = frame_imgs[frame_num]
                    objs = cls._collect_objs(spec, frame_dict, frame_img)
                    frame = Frame(spec, objs)
                    frames.append(frame)
                videos_frames.append(frames)

        else:
            start_time = time.time()
            frame_imgs = [img for frame in imgs for img in frame]
            frames = detector.detect_objs(frame_imgs)
            detector_timing += (time.time() - start_time)
            assert len(frames) == spec.num_frames * len(video_dicts), "Wrong number of frames returned"
            videos_frames = grouper(frames, spec.num_frames)

        videos = []
        for video_idx, frames in enumerate(videos_frames):
            video_dict = video_dicts[video_idx]
            questions = video_dict["questions"]
            q_types = video_dict["question_types"]
            video = Video(spec, frames)
            video.set_questions(questions, q_types)
            videos.append(video)

        return videos, detector_timing

    @classmethod
    def _collect_objs(cls, spec, frame_dict, frame_img):
        obj_dicts = frame_dict["objects"]
        objs = []
        for obj_idx, obj_dict in enumerate(obj_dicts):
            obj_type = obj_dict["class"]
            position = obj_dict["position"]
            position = tuple(map(round, position))
            colour = obj_dict["colour"]
            rotation = obj_dict["rotation"]

            # TODO Fix dataset to use external rotation value
            rotation = spec.from_internal("rotation", rotation)

            obj = Obj(spec, obj_type, position)
            obj.set_prop_val("colour", colour)
            obj.set_prop_val("rotation", rotation)
            obj.set_image(frame_img)
            objs.append(obj)

        return objs

    @classmethod
    def _collect_videos(cls, path):
        """
        Collect all videos under <path> as list of PIL images, ids and dicts
        This function executes asynchronously

        :param path: Path obj
        :return: [(id, dict, [PIL Image])]
        """

        num_workers = 16
        future_timeout = 5
        executor = ThreadPoolExecutor(max_workers=num_workers)

        futures = []
        for video_dir in path.iterdir():
            video_num = str(video_dir).split("/")[-1]
            if not video_num.isdigit():
                print(f"WARNING: {video_dir} could not be parsed into an integer")
                continue

            future = executor.submit(cls._collect_video, video_dir)
            futures.append((video_num, future))

        videos = []
        for video_num, future in futures:
            video_num = int(video_num)
            video_dict, imgs = future.result(future_timeout)
            videos.append((video_num, video_dict, imgs))

        return videos

    @classmethod
    def _collect_video(cls, video_dir):
        json_file = video_dir / "video.json"
        with json_file.open() as f:
            json_text = f.read()

        video_dict = json.loads(json_text)
        num_frames = len(video_dict["frames"])
        images = [cls._collect_img(video_dir, frame) for frame in range(num_frames)]

        return video_dict, images

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)
