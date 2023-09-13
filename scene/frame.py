import os
from scene import GaussianModel
from arguments import ModelParams
from utils.system_utils import searchForMaxIteration
from utils.camera_utils import loadCam
import numpy as np
from pathlib import Path
import torch
from PIL import Image
from utils.read_utils import remove_background_by_bounding_box, remove_background_by_mask, \
    get_2d_bounding_box, project_points, read_intrinsic_data
from utils.graphics_utils import focal2fov
from utils.pose_utils import matrix_to_quaternion
from scene.dataset_readers import CameraInfo

class Frame:
    corners = None
    root_path = None
    cam_intrinsic = None
    gaussians = None
    
    def __init__(self, id: int, args : ModelParams, gaussians : GaussianModel, load_iteration=None, resolution_scales=[1.0], cameras_extent=None, myparms=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        
        self.id = id
        self.resolution_scales = resolution_scales

        self.cameras = {}
        if Frame.root_path is None:
            Frame.root_path = Path(args.source_path)
        if Frame.corners is None:
            Frame.corners = np.loadtxt(Frame.root_path / "box3d_corners.txt")
        if Frame.cam_intrinsic is None:
            Frame.cam_intrinsic = read_intrinsic_data(Frame.root_path / "intrinsics.txt")
        if Frame.gaussians is None:
            Frame.gaussians = gaussians
        else:
            print("Gaussians already loaded, skipping")
        
        self.cameras_extent = cameras_extent
        camera = self._read_camera(
            Frame.root_path / f"poses_ba/{self.id}.txt",
            Frame.root_path / f"input/{self.id}.png",
            myparms.crop_by_bounding_box,
            myparms.crop_by_mask
        )
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.cameras[resolution_scale] = loadCam(args, self.id, camera, resolution_scale)
        
    def _read_camera(self, cam_extrinsics_file, image_path: Path, crop_by_bounding_box=False, crop_by_mask=False):

        extr = np.loadtxt(cam_extrinsics_file)[:3]
        intr = Frame.cam_intrinsic

        uid = self.id
        R = extr[:3, :3].T ##! 这里将R进行了转置
        T = extr[:3, 3]

        image = Image.open(image_path)
        image_name = f"{self.id}"

        if crop_by_bounding_box:
            points2d = project_points(extr, Frame.corners, intr)
            bounding_box_2d = get_2d_bounding_box(points2d)
            image = remove_background_by_bounding_box(image, bounding_box_2d)
        elif crop_by_mask:
            mask = np.load(image_path.parent.parent / f"mask/{self.id}.npy")
            image = remove_background_by_mask(image, mask)
        width , height = image.size

        focal_length_x = intr[0, 0]
        focal_length_y = intr[1, 1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, ## 这里图像已经加载进来了
                              image_path=image_path, image_name=image_name, width=width, height=height)
        return cam_info
    
    def transform(self, resolution, rotation=None, translation=None):
        """
        assign the pose from the camera to the gaussian points, if rotation and translation is None, we read the 
        camera extrinsics as the object pose.
        """
        assert resolution in self.resolution_scales, "resolution must be in {}".format(self.resolution_scales)
        camera = self.cameras[resolution]
        if rotation is None:
            rotation = torch.from_numpy(camera.R.T).float().cuda()
        else:
            rotation = rotation.float().cuda()
        if translation is None:
            translation = torch.from_numpy(camera.T).float().cuda()
        else:
            translation = translation.float().cuda()
        
        camera.reset_transform(np.eye(3), np.zeros(3))

        Frame.gaussians.assign_transform(rotation, translation)

        return camera
    
    @property
    def get_rotation_translation(self, resolution=1):
        camera = self.cameras[resolution]
        return torch.from_numpy(camera.R.T).float(), torch.from_numpy(camera.T).float()
    
    def get_camera(self, set_to_identity, resolution=1):
        """
        assign the pose from the camera to the gaussian points
        """
        assert resolution in self.resolution_scales, "resolution must be in {}".format(self.resolution_scales)
        camera = self.cameras[resolution]
        if set_to_identity:
            camera.reset_transform(np.eye(3), np.zeros(3))
        return camera

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        Frame.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.cameras[scale]