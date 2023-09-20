import numpy as np
import skvideo.io
import os
import glob
import tqdm
from PIL import Image
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--exp_save_root', default=".", type=str, help='视频保存路径')
parser.add_argument('--file_dir', default="debug_render", type=str,  help='包含图像的文件夹')
parser.add_argument('--max_frame', default=-1, type=int, help='最大帧数')
opt = parser.parse_args()

total_image = sorted(Path(opt.file_dir).glob("*render.png"), key=lambda x: int(x.stem.split("_")[0]))
image = np.asarray(Image.open(total_image[0]).convert("RGB"))
height, width, _ = image.shape
if opt.max_frame == -1:
	opt.max_frame = len(total_image)

for round in range(0, len(total_image) // opt.max_frame + 1):
	print(f"正在处理第{round}个视频")
	start = round * opt.max_frame
	end = (round + 1) * opt.max_frame if (round + 1) * opt.max_frame < len(total_image) else len(total_image)
	if end == start:
		break
	out_video =  np.empty([end-start, height, width, 3], dtype = np.uint8)

	for i in tqdm.trange(0, end-start):
		img =  np.asarray(Image.open(total_image[round * opt.max_frame + i]))[:, :, :3]
		out_video[i] = img

	# Writes the the output image sequences in a video file
	skvideo.io.vwrite(os.path.join(opt.exp_save_root, f"video{round}.mp4"), out_video)