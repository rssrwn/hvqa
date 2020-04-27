import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data.dataset import Dataset

import hvqa.util.func as util
from hvqa.util.environment import Video, Frame, Obj


class VideoDataset(Dataset):
    """
    Dataset for storing and fetching videos
    """

    def __init__(self, spec, data_dir, detector, hardcoded=False):
        super(VideoDataset, self).__init__()

        self.spec = spec
        self.data_dir = Path(data_dir)
        self.detector = detector
        self.hardcoded = hardcoded

        ids, self.videos, self.answers = self._find_videos()
        self.ids = {id_: idx for idx, id_ in enumerate(ids)}

    def _find_videos(self):
        video_ids = []
        videos = []
        answers = []

        print("Searching videos for data...")
        video_infos = self._collect_videos(self.data_dir)
        print(f"Found data from {len(video_infos)} videos")
        print("Extracting objects...")

        for video_num, video_dict, frame_imgs in video_infos:
            video = self._construct_video(video_dict, frame_imgs)
            ans = video_dict["answers"]
            videos.append(video)
            video_ids.append(video_num)
            answers.append(ans)

        return video_ids, videos, answers

    def _construct_video(self, video_dict, frame_imgs):
        frames = []
        frame_dicts = video_dict["frames"]
        if self.hardcoded:
            for frame_num, frame_dict in enumerate(frame_dicts):
                frame_img = frame_imgs[frame_num]
                objs = self._collect_objs(frame_dict, frame_img)
                frame = Frame(self.spec, objs)
                frame.set_objs(objs)
                frames.append(frame)
        else:
            frames = self.detector.detect_objs(frame_imgs)

        questions = video_dict["questions"]
        q_types = video_dict["question_types"]
        video = Video(self.spec, frames)
        video.set_questions(questions, q_types)
        return video

    def _collect_objs(self, frame_dict, frame_img):
        obj_dicts = frame_dict["objects"]
        objs = []
        for obj_idx, obj_dict in enumerate(obj_dicts):
            obj_type = obj_dict["class"]
            position = map(int, obj_dict["position"])
            colour = obj_dict["colour"]
            rotation = obj_dict["rotation"]
            obj = Obj(self.spec, obj_type, position)
            obj.set_prop_val("colour", colour)
            obj.set_prop_val("rotation", rotation)
            obj.set_image(frame_img)
            objs.append(obj)

        return objs

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
