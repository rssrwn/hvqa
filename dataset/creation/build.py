import json
from PIL import Image

from dataset.creation.video import Video
from dataset.creation.draw import Drawer


video_builder = Video()
video_builder.random_video()
video = video_builder.to_dict()

frame = video["frames"][0]
np_img = Drawer.draw_frame(frame)

img = Image.fromarray(np_img, "RGB")
img.save("image.png")

text = json.dumps(video)

file = open("example.txt", "w")
file.write(text)
file.close()
