import json
from PIL import Image

from dataset.creation.video import Video
from dataset.creation.draw import Drawer
from dataset.creation.definitions import *


# Create video
video_builder = Video()
video_builder.random_video()
video = video_builder.to_dict()

print("Built video")

# Write text to example file
text = json.dumps(video)
file = open("../data/example.txt", "w")
file.write(text)
file.close()

print("Written json to file")

# Create frames
for i in range(NUM_FRAMES):
    frame = video["frames"][i]
    np_img = Drawer.draw_frame(frame)
    img = Image.fromarray(np_img, "RGB")
    img.save(f"../data/frame_{i}.png")

print("Written frames to file")
