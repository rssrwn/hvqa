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

    def __init__(self, spec, data_dir, detector, hardcoded=False, group_videos=8):
        super(VideoDataset, self).__init__()

        self._detector_timing = 0

        self.spec = spec
        self.data_dir = Path(data_dir)
        self.detector = detector
        self.hardcoded = hardcoded
        self.group_videos = group_videos

        ids, self.videos, self.answers = self._find_videos()
        self.ids = {id_: idx for idx, id_ in enumerate(ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Get item of dataset

        :param item: Video number (as stored in directory)
        :return: Video obj, list of answers ([str])
        """

        idx = self.ids[item]
        video = self.videos[idx]
        ans = self.answers[idx]
        return video, ans

    def is_hardcoded(self):
        return self.hardcoded

    def detector_timing(self):
        return self._detector_timing

    def _find_videos(self):
        video_ids = []
        videos = []
        answers = []

        print("Searching videos for data...")
        video_infos = self._collect_videos(self.data_dir)
        print(f"Found data from {len(video_infos)} videos")
        print("Extracting objects...")

        # Group videos together for faster object detection
        for video_info in grouper(video_infos, self.group_videos):
            video_info = [info for info in video_info if info is not None]
            video_nums, video_dicts, frame_imgs = tuple(zip(*video_info))
            grouped_videos = self._construct_videos(video_dicts, frame_imgs)
            videos.extend(grouped_videos)
            video_ids.extend(video_nums)
            ans = [video_dict["answers"] for video_dict in video_dicts]
            answers.extend(ans)

        return video_ids, videos, answers

    def _construct_videos(self, video_dicts, imgs):
        if self.hardcoded:
            videos_frames = []
            for idx, video_dict in enumerate(video_dicts):
                frame_imgs = imgs[idx]
                frame_dicts = video_dict["frames"]
                frames = []
                for frame_num, frame_dict in enumerate(frame_dicts):
                    frame_img = frame_imgs[frame_num]
                    objs = self._collect_objs(frame_dict, frame_img)
                    frame = Frame(self.spec, objs)
                    frames.append(frame)
                videos_frames.append(frames)

        else:
            start_time = time.time()
            frame_imgs = [img for frame in imgs for img in frame]
            frames = self.detector.detect_objs(frame_imgs)
            self._detector_timing += (time.time() - start_time)
            assert len(frames) == self.spec.num_frames * len(video_dicts), "Wrong number of frames returned"
            videos_frames = grouper(frames, self.spec.num_frames)

        videos = []
        for video_idx, frames in enumerate(videos_frames):
            video_dict = video_dicts[video_idx]
            questions = video_dict["questions"]
            q_types = video_dict["question_types"]
            video = Video(self.spec, frames)
            video.set_questions(questions, q_types)
            videos.append(video)

        return videos

    def _collect_objs(self, frame_dict, frame_img):
        obj_dicts = frame_dict["objects"]
        objs = []
        for obj_idx, obj_dict in enumerate(obj_dicts):
            obj_type = obj_dict["class"]
            position = obj_dict["position"]
            position = tuple(map(round, position))
            colour = obj_dict["colour"]
            rotation = obj_dict["rotation"]

            # TODO Fix dataset to use external rotation value
            rotation = self.spec.from_internal("rotation", rotation)

            obj = Obj(self.spec, obj_type, position)
            obj.set_prop_val("colour", colour)
            obj.set_prop_val("rotation", rotation)
            obj.set_image(frame_img)
            objs.append(obj)

        return objs

    def _collect_videos(self, path):
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

            future = executor.submit(self._collect_video, video_dir, video_num)
            futures.append((video_num, future))

        videos = []
        for video_num, future in futures:
            video_num = int(video_num)
            video_dict, imgs = future.result(future_timeout)
            videos.append((video_num, video_dict, imgs))

        return videos

    def _collect_video(self, video_dir, video_num):
        json_file = video_dir / "video.json"
        with json_file.open() as f:
            json_text = f.read()

        video_dict = json.loads(json_text)
        num_frames = len(video_dict["frames"])
        images = [self._collect_img(self.data_dir / str(video_num), frame) for frame in range(num_frames)]

        return video_dict, images

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)
