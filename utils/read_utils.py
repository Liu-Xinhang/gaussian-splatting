import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Union

def read_intrinsic_data(intrinsic_path: Union[str, Path]) -> np.ndarray:
	with open(intrinsic_path, 'r') as file:
		data = file.readlines()
	## handle intrinsics.txt
	intrinsic_dict = {}
	for item in data:
		name, value = item.split(":")
		intrinsic_dict[name] = float(value.strip())
	intrinsic = np.array([
		[intrinsic_dict["fx"], 0, intrinsic_dict["cx"]], 
		[0, intrinsic_dict["fy"], intrinsic_dict["cy"]], 
		[0, 0, 1]])
	return intrinsic

def project_points(pose, points, intrinsic):
        points = np.concatenate((points, np.ones((points.shape[0], 1))), -1).T # 4, N
        points2d = intrinsic @ pose @ points
        points2d = points2d[:2] / points2d[2]
        return points2d.T

def get_2d_bounding_box(bounding_box_3d):
    left, top = np.min(bounding_box_3d, axis=0)
    right, bottom = np.max(bounding_box_3d, axis=0)
    return np.array([left, top, right, bottom], dtype=np.int32)

def remove_background_by_bounding_box(img, bounding_box_2d):
    # 创建一个ImageDraw对象
    draw = ImageDraw.Draw(img)
    # 定义矩形框的坐标(左上角(x0, y0)和右下角(x1, y1))
    width, height = img.size
    x0, y0, x1, y1 = bounding_box_2d
    # 在黑色图片上绘制一个白色的矩形
    # draw.rectangle([x0, y0, x1, y1], fill='white')
    draw.rectangle([0, 0, width, y0], fill='black') # 上方区域
    draw.rectangle([0, y1, width, height], fill='black') # 下方区域
    draw.rectangle([0, y0, x0, y1], fill='black') # 左侧区域
    draw.rectangle([x1, y0, width, y1], fill='black') # 右侧区域
    return img

def remove_background_by_mask(img, mask):
    img = np.array(img)
    img[~mask] = 0
    return Image.fromarray(img)