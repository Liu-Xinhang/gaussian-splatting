import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="temp_datasets/loquat-2")
args = parser.parse_args()

vidcap = cv2.VideoCapture(os.path.join(args.root_dir, "Frames.m4v"))
success,image = vidcap.read()
count = 0
root_dir = os.path.join(args.root_dir, "input")
os.makedirs(root_dir, exist_ok=True)
while success:
  cv2.imwrite(f"{root_dir}/{count}.png", image)     # save frame as PNG file      
  success,image = vidcap.read()
  count += 1