import cv2
import os
vidcap = cv2.VideoCapture('/home/liuxinhang/datasets_temp/val_data/0601-loquat-box/loquat-1/Frames.m4v')
success,image = vidcap.read()
count = 0
root_dir = "/home/liuxinhang/datasets_temp/val_data/0601-loquat-box/loquat-1/color_full"
os.makedirs(root_dir, exist_ok=True)
while success:

  cv2.imwrite(f"{root_dir}/{count}.png", image)     # save frame as PNG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1