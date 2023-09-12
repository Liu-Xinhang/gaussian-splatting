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
    

    def __init__(self, id: int, args : ModelParams, gaussians : GaussianModel, load_iteration=None, resolution_scales=[1.0], cameras_extent=None, myparms=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = None
        
        self.id = id
        self.resolution_scales = resolution_scales

        self.cameras = {}
        if Frame.root_path is None:
            Frame.root_path = Path(args.source_path)
        if Frame.corners is None:
            Frame.corners = np.loadtxt(Frame.root_path / "box3d_corners.txt")
        if Frame.cam_intrinsic is None:
            Frame.cam_intrinsic = read_intrinsic_data(Frame.root_path / "intrinsics.txt")

        self.gaussians = gaussians
        assert load_iteration is not None, "load_iteration must be specified"
        if load_iteration == -1:  ## 加载训练好的场景，如果是-1，那么就是最新的场景
            self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
        else:
            self.loaded_iter = load_iteration
        print("Loading trained model at iteration {}".format(self.loaded_iter))
        self.gaussians.load_ply(os.path.join(self.model_path,
                            "point_cloud",
                            "iteration_" + str(self.loaded_iter),
                            "point_cloud.ply"))


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
        R = extr[:3, :3].T
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
    
    def transform(self, resolution):
        """
        assign the pose from the camera to the gaussian points
        """
        assert resolution in self.resolution_scales, "resolution must be in {}".format(self.resolution_scales)
        camera = self.cameras[resolution]
        rotation = torch.from_numpy(camera.R.T).float().cuda()
        translation = torch.from_numpy(camera.T).float().cuda()

        camera.reset_transform(np.eye(3), np.zeros(3))

        self.gaussians.assign_transform(rotation, translation)
        return camera

    def get_camera(self, resolution):
        """
        assign the pose from the camera to the gaussian points
        """
        assert resolution in self.resolution_scales, "resolution must be in {}".format(self.resolution_scales)
        camera = self.cameras[resolution]
        return camera

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.cameras[scale]