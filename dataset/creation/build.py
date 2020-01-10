import json

from dataset.creation.video import Video


video_builder = Video()
video_builder.random_video()
video = video_builder.to_dict()

print(video)
text = json.dumps(video)

file = open("example.txt", "w")
file.write(text)
file.close()
