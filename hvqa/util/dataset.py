import json
from pathlib import Path
from torch.utils.data.dataset import Dataset

import hvqa.util.func as util
from hvqa.util.environment import Video, Frame, Obj


class VideoDataset(Dataset):
    """
    Dataset for storing and fetching videos
    """

    def __init__(self, spec, data_dir, hardcoded=False):
        super(VideoDataset, self).__init__()

        self.spec = spec
        self.data_dir = Path(data_dir)
        self.hardcoded = hardcoded

        ids, self.videos = self._find_videos()
        self.ids = {id_: idx for idx, id_ in enumerate(ids)}

    def _find_videos(self):
        video_dirs = self.data_dir.iterdir()

        video_ids = []
        videos = []

        num_videos = 0

        print("Searching videos for data...")

        # Iterate through videos
        for video_dir in video_dirs:
            video_num = str(video_dir).split("/")[-1]
            if not video_num.isdigit():
                print(f"WARNING: {video_dir} could not be parsed into an integer")
                continue

            video_num = int(video_num)

            json_file = video_dir / "video.json"
            with json_file.open() as f:
                json_text = f.read()

            video_dict = json.loads(json_text)
            frame_dicts = video_dict["frames"]

            # Iterate through frames in current video
            frames = []
            for frame_num, frame_dict in enumerate(frame_dicts):
                frame_img = self._collect_img(self.data_dir / str(video_num), frame_num)
                objs = self._collect_objs(frame_dict, frame_img)
                frame = Frame(self.spec, frame_img)
                frame.set_objs(objs) if objs is not None else None
                frames.append(frame)

            video = Video(self.spec, frames)
            questions = video_dict["questions"]
            q_types = video_dict["question_types"]
            video.set_questions(questions, q_types)
            videos.append(video)
            video_ids.append(video_num)
            num_videos += 1

        print(f"Found data from {num_videos} videos")

        return video_ids, videos

    def _collect_objs(self, frame_dict, frame_img):
        obj_dicts = frame_dict["objects"]
        if self.hardcoded:
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
        else:
            objs = None

        return objs

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Get item of dataset

        :param item: Video number (as stored in directory)
        :return: Video obj
        """

        idx = self.ids[item]
        video = self.videos[idx]
        return video

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)
