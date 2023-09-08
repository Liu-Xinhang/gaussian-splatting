## run in sam environment
## 这个文件将根据onepose数据集中的内容生成mask。思路是根据3d bounding box
## 生成2d bounding box，然后用sam生成mask
## 注意，应该提前将视频转换为图像，然后通过
from pathlib import Path
import numpy as np
import cv2
from segment_anything import SamPredictor, sam_model_registry
from matplotlib import pyplot as plt
import tqdm
import argparse

def project_points(pose, points, intrinsic):
    points = np.concatenate((points, np.ones((points.shape[0], 1))), -1).T # 4, N
    points2d = intrinsic @ pose @ points
    points2d = points2d[:2] / points2d[2]
    return points2d.T

def read_intrinsic_data(intrinsic_path: str) -> np.ndarray:
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

def get_2d_bounding_box(bounding_box_3d):
    left, top = np.min(bounding_box_3d, axis=0)
    right, bottom = np.max(bounding_box_3d, axis=0)
    return np.array([left, top, right, bottom], dtype=np.int32)

def draw_bounding_box(bounding_box_2d, image, color):
    left, top, right, bottom = bounding_box_2d
    points_uv = np.array([[left, top], [right, top], [right, bottom], [left, bottom]])
    image = cv2.line(image, points_uv[0], points_uv[1], color, 2)
    image = cv2.line(image, points_uv[1], points_uv[2], color, 2)
    image = cv2.line(image, points_uv[3], points_uv[2], color, 2)
    image = cv2.line(image, points_uv[3], points_uv[0], color, 2)
    return image

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="temp_datasets/loquat-2")
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)

    sam_model_path = "/home/liuxinhang/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_model_path).cuda()
    predictor = SamPredictor(sam)

    ## read intrinsic matrix
    intrinsic = read_intrinsic_data(str(root_dir / "intrinsics.txt"))
    
    ## read all poses file
    pose_files = (root_dir / "poses_ba").glob("*.txt")
    pose_files = sorted(pose_files, key=lambda x: int(x.stem))

    # read corner information to generate the 2d bounding box
    box3d_path = root_dir / "box3d_corners.txt"
    box3d = np.loadtxt(box3d_path)

    save_vis_dir = root_dir / "masks_vis"
    save_vis_dir.mkdir(exist_ok=True)
    save_dir = root_dir / "mask"
    save_dir.mkdir(exist_ok=True)
    with tqdm.tqdm(range(len(pose_files))) as pbar:
        for pose_file in pose_files:
            pose = np.loadtxt(pose_file)[:3]
            points2d = project_points(pose, box3d, intrinsic)
            bounding_box_2d = get_2d_bounding_box(points2d)

            image_path = str(pose_file).replace("poses_ba", "input").replace(".txt", ".png")
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictor.set_image(image)

            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bounding_box_2d[None, :],
                multimask_output=False,
            )
            np.save(save_dir / f"{pose_file.stem}.npy", masks[0])
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(masks[0], plt.gca())
            show_box(bounding_box_2d, plt.gca())
            plt.axis('off')
            plt.savefig(save_vis_dir / f"{pose_file.stem}.png")
            plt.close()
            pbar.update()