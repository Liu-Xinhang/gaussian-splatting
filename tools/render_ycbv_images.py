"""
This file is used to render the ycbv image
"""
import sys
import os
script_path = os.path.abspath(__file__)
# 获取脚本所在目录的父目录，即 gaussian-splatting 的路径
gaussian_splatting_dir = os.path.dirname(os.path.dirname(script_path))
# 将这个目录添加到 sys.path
sys.path.append(gaussian_splatting_dir)

from utils.render_utils import Renderer
import json
import torch
from torchvision import utils
from scipy.spatial.transform import Rotation as R
from itertools import product
import pathlib
import numpy as np
import argparse
import tqdm

class ReferenceImageGenerator:
    def __init__(self, renderer: dict, K):
        self.renderer = Renderer(**renderer)
        self.K = K
    
    def to(self, device):
        self.renderer.to(device) ## 把pytorch3d移到指定设备上
        self.K = self.K.to(device)


    def render_object(self, rotation, translation, obj_number):
        rotation = rotation.unsqueeze(0)
        translation = translation.unsqueeze(0)
        k = self.K.unsqueeze(0)
        render_outputs = self.renderer(rotation, translation, k, torch.tensor([obj_number]))
        rendered_images, rendered_fragments = render_outputs['images'], render_outputs['fragments']
        rendered_images = rendered_images[..., :3].permute(0, 3, 1, 2).contiguous()
        rendered_depths = rendered_fragments.zbuf
        rendered_depths = rendered_depths[..., 0]
        # rendered_masks = (rendered_depths > 0).to(torch.float32)
        return rendered_images, rendered_depths


if __name__ == "__main__":
    dataset_root = 'temp_datasets/ycbv'
    image_scale = 256
    device = torch.device('cuda:0')

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    args = parser.parse_args()


    renderer=dict(
        mesh_dir=dataset_root + f'/models_obj/obj_{args.id:06d}.obj',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(1, 1, 1),
    )

    save_root_dir = pathlib.Path("reference_images") / f"{args.id:06d}"
    save_root_dir.mkdir(parents=True, exist_ok=True)

    K = torch.tensor([
        [1066.0, 0, 128],
        [0, 1066, 128], 
        [0, 0, 1]
    ])


    ## read model diameter
    with open("temp_datasets/ycbv/models_eval/models_info.json") as file:
        model_info = json.load(file)
    diameter = model_info[str(args.id)]["diameter"]
    print("diameter(mm): ", diameter)

    ## save K
    np.savetxt(save_root_dir / "intrinsic.txt", K)

    reference_image_generator = ReferenceImageGenerator(renderer, K)
    reference_image_generator.to(device)
    
    ## 生成rotation
    X_directions = np.arange(0, 180+0.00001, 22.5/2)
    Z_directions = np.arange(0, 360, 22.5/2)
    
    image_saving_dir = save_root_dir / "images"
    image_saving_dir.mkdir(parents=True, exist_ok=True)
    pose_saving_dir =save_root_dir / "poses"
    pose_saving_dir.mkdir(parents=True, exist_ok=True)
    # depth_saving_dir = pathlib.Path('reference_images/depths')
    # depth_saving_dir.mkdir(parents=True, exist_ok=True)
    translation = torch.tensor([0, 0, 4.36 * diameter], device=device) # 4.36 is a experience number
    with tqdm.tqdm(range(len(X_directions) * len(Z_directions))) as pbar:
        for id, (x_direction, z_direction) in enumerate(product(X_directions, Z_directions)):
            rotation = R.from_euler("zyx", [z_direction, 0, x_direction], degrees=True) # for z direction, we simply omit it for the symmetry
            pose = np.concatenate([rotation.as_matrix(), translation.cpu().numpy().reshape(3, 1)], axis=1)
            np.save(pose_saving_dir / f'{id:06d}.npy', pose)
            rotation = torch.from_numpy(rotation.as_matrix()).to(device).float()
            image, depth = reference_image_generator.render_object(rotation, translation, args.id)
            image_saving_path = image_saving_dir / f'{id:06d}.png'
            # depth_saving_path = depth_saving_dir / f'{id:06d}.pt'
            utils.save_image(image, image_saving_path)
            # torch.save(depth, depth_saving_path)
            pbar.update(1)
    