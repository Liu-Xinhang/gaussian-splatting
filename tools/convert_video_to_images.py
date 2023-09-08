import cv2
import os
vidcap = cv2.VideoCapture('temp_datasets/loquat-2/Frames.m4v')
success,image = vidcap.read()
count = 0
root_dir = "temp_datasets/loquat-2/input"
os.makedirs(root_dir, exist_ok=True)
while success:

  cv2.imwrite(f"{root_dir}/{count}.png", image)     # save frame as PNG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1