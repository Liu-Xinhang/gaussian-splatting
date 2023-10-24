#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image, ImageDraw
import random
import pickle
from typing import NamedTuple, Union, Sequence
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.geometry_utils import load_ply, select_poses
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.read_utils import remove_background_by_bounding_box, remove_background_by_mask, \
    get_2d_bounding_box, project_points, read_intrinsic_data
from utils.load_llff import load_llff_data
from tqdm import tqdm
from typing import Optional
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: Optional[np.array] = None
    mask_vis: Optional[np.ndarray] = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info): 
    def get_center_and_diag(cam_centers): ##Todo 这是在干什么
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readYCBVBOPCameras(results):
    cam_infos = []
    print("total files: ", len(results))
    for idx, result in enumerate(results):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(results)))
        sys.stdout.flush()

        extr = result["pose"]
        extr[:3, 3] /= 1000 # convert mm to m
        intr = result["cam_K"]

        uid = idx
        R = extr[:3, :3].T
        T = extr[:3, 3]

        image_path = result["img_path"]
        image_name = f"{idx}"
        mask_path = result["mask_path"]
        mask = np.array(Image.open(mask_path))
        mask = mask == 255
        image = Image.open(image_path)
        image = remove_background_by_mask(image, mask)

        width , height = image.size

        focal_length_x = intr[0, 0]
        focal_length_y = intr[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readYCBVRenderCameras(cam_extrinsics_files, cam_intrinsic, images_folder: Path):
    cam_infos = []
    for idx, cam_extrinsics_file in enumerate(cam_extrinsics_files):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics_files)))
        sys.stdout.flush()

        extr = np.load(cam_extrinsics_file)[:3]
        extr[:3, 3] /= 1000 # convert mm to m
        intr = cam_intrinsic

        uid = idx
        R = extr[:3, :3].T
        T = extr[:3, 3]

        image_path = images_folder / f"{idx:06d}.png"
        image_name = f"{idx}"
        image = Image.open(image_path)

        width , height = image.size

        focal_length_x = intr[0, 0]
        focal_length_y = intr[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def readOneposeCameras(cam_extrinsics_files, cam_intrinsic, images_folder:Path, crop_by_bounding_box=False, crop_by_mask=False):

    ## 加载点，为crop_by_boundingbox 做准备
    corners = np.loadtxt(images_folder.parent.parent / "box3d_corners.txt")
    ## 加载mask，为crop_by_mask 做准备
    total_mask_paths = (images_folder.parent / "mask").glob("*.npy")
    total_mask_paths = sorted(total_mask_paths, key=lambda x: int(x.stem))

    cam_infos = []
    for idx, cam_extrinsics_file in enumerate(cam_extrinsics_files):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics_files)))
        sys.stdout.flush()

        extr = np.loadtxt(cam_extrinsics_file)[:3]
        intr = cam_intrinsic

        uid = idx
        R = extr[:3, :3].T
        T = extr[:3, 3]

        image_path = images_folder / f"{idx}.png"
        image_name = f"{idx}"
        image = Image.open(image_path)

        if crop_by_bounding_box:
            points2d = project_points(extr, corners, intr)
            bounding_box_2d = get_2d_bounding_box(points2d)
            image = remove_background_by_bounding_box(image, bounding_box_2d)
        elif crop_by_mask:
            mask = np.load(total_mask_paths[idx])
            assert idx == int(total_mask_paths[idx].stem), "mask file name is not consistent with the camera index"
            image = remove_background_by_mask(image, mask)
        width , height = image.size

        focal_length_x = intr[0, 0]
        focal_length_y = intr[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def readllffCameras(poses, images):
    cam_infos = []
    for idx, (image, pose) in enumerate(zip(images, poses)):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(poses)))
        sys.stdout.flush()

        c2w = pose[:3, :4]
        c2w[:3, 1:3] *= -1
        c2w = np.concatenate([c2w, np.array([[0, 0, 0, 1]])], axis=0) ## 4 * 4
        w2c = np.linalg.inv(c2w)
        uid = idx
        R = w2c[:3, :3].T
        T = w2c[:3, 3]

        image_path = f"{idx}"
        image_name = f"{idx}"

        height, width = image.shape[:2]
        image = Image.fromarray(np.uint8(image * 255))

        focal_length_x = pose[2, 4]
        focal_length_y = focal_length_x
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width) ## 视场角
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readYCBVBOPInfo(path, eval, resolution, category_type, obj_number, llffhold=8):
    def generate_xyz_and_rangdom_rgb(model_path, resolution: int):
        xyz = load_ply(model_path) / 1000 # N, 3 convert mm to m
        if resolution > xyz.shape[0]:
            print(f"resolution {resolution} is larger than the number of points in the model ({xyz.shape[0]}), we will use all the points.")
        else:
            selected_rows = np.random.choice(xyz.shape[0], resolution, replace=False)
            xyz = xyz[selected_rows]
        rgb = np.random.random((xyz.shape[0], 3))
        return xyz, rgb
    
    def grap_all_obj_image(path, obj_number, profile="png"):
        path = Path(path)
        total_scene_gt = sorted(path.rglob("scene_gt.json"))
        total_scene_gt_info = sorted(path.rglob("scene_gt_info.json"))
        total_scene_camera = sorted(path.rglob("scene_camera.json"))
        ## category_type: pbr, real, synt
        res = []
        with tqdm(total=len(total_scene_gt)) as pbar:
            for scene_gt_path, scene_gt_info_path, scene_camera_path in zip(total_scene_gt, total_scene_gt_info, total_scene_camera):
                scene_gt = json.load(open(scene_gt_path))
                scene_gt_info = json.load(open(scene_gt_info_path))
                scene_camera = json.load(open(scene_camera_path))
                data = {}
                for image_number, total_objs in scene_gt.items():
                    for i, obj in enumerate(total_objs):
                        if obj["obj_id"] == obj_number:
                            if  scene_gt_info[image_number][i]["visib_fract"] > 0.5: ## 至少要有0.5的可见区域
                                image_path = scene_gt_path.parent / "rgb" / f"{int(image_number):06d}.{profile}"
                                data["img_path"] = image_path
                                data["mask_path"] = scene_gt_path.parent / "mask_visib" / f"{int(image_number):06d}_{i:06d}.png"
                                data["cam_K"] = np.array(scene_camera[image_number]["cam_K"]).reshape(3, 3)
                                R = np.array(obj["cam_R_m2c"]).reshape(3, 3)
                                T = np.array(obj["cam_t_m2c"]).reshape(3, 1)
                                data["pose"] = np.concatenate([R, T], axis=1)
                                res.append(data.copy())
                pbar.update()
        return res
    
    path = Path(path)
    item_number = 0
    if "pbr" in category_type:
        item_number += 4
    if "real" in category_type:
        item_number += 2
    if "synt" in category_type:
        item_number += 1
    if os.path.exists("cache/{obj_number:06d}_{item_number}.pkl"):
        print("find cache file")
        results = pickle.load(open(f"cache/{obj_number:06d}_{item_number}.pkl", "rb"))
    else:
        print("no catch file found")
        results = []
        if "pbr" in category_type:
            results.extend(grap_all_obj_image("temp_datasets/ycbv/train_pbr", obj_number, "jpg"))
        if "real" in category_type:
            results.extend(grap_all_obj_image("temp_datasets/ycbv/train_real", obj_number))
        if "synt" in category_type:
            results.extend(grap_all_obj_image("temp_datasets/ycbv/train_synt", obj_number))
    
        ## subsample results
        # results = random.sample(results, 700)
        results = select_poses(results, 700, 10)
        pickle.dump(results, open(f"cache/{obj_number:06d}_{item_number}.pkl", "wb"))

    cam_infos = readYCBVBOPCameras(results)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    model_path = Path("temp_datasets") / "ycbv" / "models_eval" / f"obj_{obj_number:06d}.ply"
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    if True or not os.path.exists(ply_path): ##!debug阶段每次都重新生成
        ## generate model from obj_files
        xyz, rgb, = generate_xyz_and_rangdom_rgb(model_path, resolution)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readYCBVRenderInfo(path, eval, resolution, llffhold=8):
    def generate_xyz_and_rangdom_rgb(model_path, resolution: int):
        xyz = load_ply(model_path) / 1000 # N, 3 convert mm to m
        if resolution > xyz.shape[0]:
            print(f"resolution {resolution} is larger than the number of points in the model ({xyz.shape[0]}), we will use all the points.")
        else:
            selected_rows = np.random.choice(xyz.shape[0], resolution, replace=False)
            xyz = xyz[selected_rows]
        rgb = np.random.random((xyz.shape[0], 3))
        return xyz, rgb

    cam_extrinsics_root = Path(path) / "poses"
    cam_extrinsics_files = cam_extrinsics_root.glob("*.npy")
    cam_extrinsics_files = sorted(cam_extrinsics_files, key=lambda x: int(x.stem))
    cam_intrinsics_path = Path(path) / "intrinsic.txt"
    image_path = Path(path) / "images"
    cam_intrinsic = np.loadtxt(cam_intrinsics_path)

    cam_infos = readYCBVRenderCameras(cam_extrinsics_files=cam_extrinsics_files, cam_intrinsic=cam_intrinsic, images_folder=image_path)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    model_path = Path("temp_datasets") / "ycbv" / "models_eval" / f"obj_{int(Path(path).stem):06d}.ply"
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    if True or not os.path.exists(ply_path): ##!debug阶段每次都重新生成
        ## generate model from obj_files
        xyz, rgb, = generate_xyz_and_rangdom_rgb(model_path, resolution)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info
    
def readOneposeInfo(path, eval, resolution=[40, 80, 40], llffhold=8, crop_by_bounding_box=False, crop_by_mask=False, volume_init=False): ## for onepose dataset
    def generate_random_xyz_and_rgb(bounding_box_path, resolution: Union[int, Sequence[int]], volume_init=False):
        def sample_points_from_volume(bounding_box: np.ndarray, resolution: Union[int, Sequence[int]]):
            if isinstance(resolution, int):
                resolution = [resolution, resolution, resolution]
            assert len(resolution) == 3, "resolution must be a sequence of length 3"
            x_min, y_min, z_min = bounding_box[0]
            x_max, y_max, z_max = bounding_box[6]
            x = np.linspace(x_min, x_max, resolution[0])
            y = np.linspace(y_min, y_max, resolution[1])
            z = np.linspace(z_min, z_max, resolution[2])
            xx, yy, zz = np.meshgrid(x, y, z)
            return np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        def sample_points_from_bounding_box(bounding_box: np.ndarray, resolution: Union[int, Sequence[int]]):
            if isinstance(resolution, int):
                resolution = [resolution, resolution, resolution]
            assert len(resolution) == 3, "resolution must be a sequence of length 3"
            # 定义6个面
            faces = [
                [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
                [bounding_box[4], bounding_box[5], bounding_box[6], bounding_box[7]],
                [bounding_box[0], bounding_box[1], bounding_box[5], bounding_box[4]],
                [bounding_box[1], bounding_box[2], bounding_box[6], bounding_box[5]],
                [bounding_box[2], bounding_box[3], bounding_box[7], bounding_box[6]],
                [bounding_box[3], bounding_box[0], bounding_box[4], bounding_box[7]]
            ]
            total_sample_points = []
            # 进行采样
            for face in faces:
                for i in np.linspace(0, 1, resolution):
                    for j in np.linspace(0, 1, resolution):
                        # 计算采样点的坐标
                        sample_point = (
                            (1 - i) * (1 - j) * np.array(face[0]) +
                            i * (1 - j) * np.array(face[1]) +
                            i * j * np.array(face[2]) +
                            (1 - i) * j * np.array(face[3])
                        )
                        total_sample_points.append(sample_point)
            return np.array(total_sample_points)
        bounding_box = np.loadtxt(bounding_box_path)
        if volume_init:
            xyz = sample_points_from_volume(bounding_box, resolution)
        else:
            xyz = sample_points_from_bounding_box(bounding_box, resolution)
        rgb = np.random.random((xyz.shape[0], 3))
        return xyz, rgb
    
    cam_extrinsics_root = Path(path) / "poses_ba"
    cam_extrinsics_files = cam_extrinsics_root.glob("*.txt")
    cam_extrinsics_files = sorted(cam_extrinsics_files, key=lambda x: int(x.stem))
    cam_intrinsics_path = Path(path) / "intrinsics.txt"
    image_path = Path(path) / "input"
    cam_intrinsic = read_intrinsic_data(str(cam_intrinsics_path))
    cam_infos = readOneposeCameras(cam_extrinsics_files=cam_extrinsics_files, cam_intrinsic=cam_intrinsic, images_folder=image_path, crop_by_bounding_box=crop_by_bounding_box, crop_by_mask=crop_by_mask)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bounding_box_path = Path(path).parent / "box3d_corners.txt"
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    if True or not os.path.exists(ply_path): ##!debug阶段每次都重新生成
        print("random generate point cloud from the bounding box, will happen only the first time you open the scene.")
        xyz, rgb, = generate_random_xyz_and_rgb(bounding_box_path, resolution, volume_init)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readLLFFInfo(path, eval, llffhold=8):
    def generate_random_xyz_and_rgb(resolution: Union[int, Sequence[int]], volume_init=True):
        def sample_points_from_volume(bounding_box: np.ndarray, resolution: Union[int, Sequence[int]]):
            if isinstance(resolution, int):
                resolution = [resolution, resolution, resolution]
            assert len(resolution) == 3, "resolution must be a sequence of length 3"
            x_min, y_min, z_min = bounding_box[0]
            x_max, y_max, z_max = bounding_box[6]
            x = np.linspace(x_min, x_max, resolution[0])
            y = np.linspace(y_min, y_max, resolution[1])
            z = np.linspace(z_min, z_max, resolution[2])
            xx, yy, zz = np.meshgrid(x, y, z)
            return np.vstack([xx.flatten(), yy.flatten(), zz.flatten()]).T
        def sample_points_from_bounding_box(bounding_box:np.ndarray, resolution: int):
            assert isinstance(resolution, int), "resolution must be an integer"
            # 定义6个面
            faces = [
                [bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]],
                [bounding_box[4], bounding_box[5], bounding_box[6], bounding_box[7]],
                [bounding_box[0], bounding_box[1], bounding_box[5], bounding_box[4]],
                [bounding_box[1], bounding_box[2], bounding_box[6], bounding_box[5]],
                [bounding_box[2], bounding_box[3], bounding_box[7], bounding_box[6]],
                [bounding_box[3], bounding_box[0], bounding_box[4], bounding_box[7]]
            ]
            total_sample_points = []
            # 进行采样
            for face in faces:
                for i in np.linspace(0, 1, resolution):
                    for j in np.linspace(0, 1, resolution):
                        # 计算采样点的坐标
                        sample_point = (
                            (1 - i) * (1 - j) * np.array(face[0]) +
                            i * (1 - j) * np.array(face[1]) +
                            i * j * np.array(face[2]) +
                            (1 - i) * j * np.array(face[3])
                        )
                        total_sample_points.append(sample_point)
            return np.array(total_sample_points)
        bounding_box = np.array(
            [[-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, 1],
            [1, 1, -1]]
        )
        if volume_init:
            xyz = sample_points_from_volume(bounding_box, resolution)
        else:
            xyz = sample_points_from_bounding_box(bounding_box, resolution)
        rgb = np.random.random((xyz.shape[0], 3))
        return xyz, rgb
    
    images, poses, bds, render_poses, i_test = load_llff_data(path)
    cam_infos_unsorted = readllffCameras(poses, images)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        print("eval mode activate   ")
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)

    if True or not os.path.exists(ply_path): ##!debug阶段每次都重新生成
        print("random generate point cloud from the bounding box, will happen only the first time you open the scene.")
        xyz, rgb, = generate_random_xyz_and_rgb(40)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Onepose" : readOneposeInfo,
    "llff" : readLLFFInfo,
    "ycbv_render": readYCBVRenderInfo,
    "ycbv_bop": readYCBVBOPInfo
}