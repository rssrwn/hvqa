import json
from pathlib import Path
from torch.utils.data.dataset import Dataset

import hvqa.util as util


class VideoDataset(Dataset):
    """
    Dataset for storing and fetching videos
    """

    def __init__(self, data_dir):
        super(VideoDataset, self).__init__()

        self.data_dir = Path(data_dir)
        self.ids, self.videos = self._find_videos()

    def _find_videos(self):
        video_dirs = self.data_dir.iterdir()

        video_ids = []
        videos = []

        num_videos = 0

        print("Searching videos for data...")

        # Iterate through videos
        for video_dir in video_dirs:
            video_num = int(str(video_dir).split("/")[-1])
            json_file = video_dir / "video.json"
            if json_file.exists():
                with json_file.open() as f:
                    json_text = f.read()

                video_dict = json.loads(json_text)
                videos.append(video_dict)
                video_ids.append(video_num)

                num_videos += 1

        print(f"Found data from {num_videos} videos")

        return video_ids, videos

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
        Get item of dataset

        :param item: Index of video in the dataset
        :return: List of PIL Images, video_dict
        """

        id = self.ids[item]
        video_dir = self.data_dir / str(id)
        video_dict = self.videos[item]
        imgs = [self._collect_img(video_dir, idx) for idx in range(len(video_dict["frames"]))]
        return imgs, video_dict

    @staticmethod
    def _collect_img(video_dir, frame_idx):
        return util.collect_img(video_dir, frame_idx)
