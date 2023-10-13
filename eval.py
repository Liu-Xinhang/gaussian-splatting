from argparse import ArgumentParser, Namespace
from scene import GaussianModel
from scene.frame import OneposeFrame, NeRFFrame, llffFrame
import sys
from arguments import ModelParams, OptimizationParams, MyParams, PipelineParams, OptimizierParams
from itertools import product
from gaussian_renderer import render
from scipy.spatial.transform import Rotation as R
import torch
import torchvision
from pathlib import Path
import numpy as np
import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import matrix_to_quaternion, euler_angles_to_matrix, quaternion_to_matrix
import os
from utils.system_utils import searchForMaxIteration
from utils.geometry_utils import load_ply, calculate_models_diameter
from utils.eval_utils import DegreeAndCM


def eval(image_id, dataset, op, pipe, load_iteration, myparms, opt, init_translation=None, init_rotation=None, pose_disturbance=None, comment=None):
    gaussians = GaussianModel(dataset.sh_degree)

    ## load gaussian
    assert load_iteration is not None, "load_iteration must be specified"
    if load_iteration == -1:  ## 加载训练好的场景，如果是-1，那么就是最新的场景
        loaded_iter = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    else:
        loaded_iter = load_iteration
    print("Loading trained model at iteration {}".format(loaded_iter))
    gaussians.load_ply(os.path.join(dataset.model_path,
                        "point_cloud",
                        "iteration_" + str(loaded_iter),
                        "point_cloud.ply"))
    gaussians.load_camera_extent(os.path.join(dataset.model_path,
                        "point_cloud",
                        "iteration_" + str(loaded_iter),
                        "cameras_extent.json"
                        ))
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if myparms.mytype=="Onepose":
        frame = OneposeFrame(image_id, dataset, gaussians, load_iteration, cameras_extent=0.5027918756008148, myparms=myparms)
    elif myparms.mytype=="Nerf":
        frame = NeRFFrame(image_id, dataset, gaussians, myparms=myparms)
    elif myparms.mytype=="llff":
        frame = llffFrame(image_id, dataset, gaussians)
    else:
        raise NotImplementedError
    obj_name = Path(dataset.source_path).stem
    save_dir = Path(f"debug_data_{obj_name}")
    save_dir.mkdir(exist_ok=True)

    evler = DegreeAndCM(translation_scale="m")

    rotation, translation = frame.get_rotation_translation()
    gt_pose = torch.cat((rotation, translation[:, None]), -1)
    rotation = matrix_to_quaternion(rotation)
    if pose_disturbance is not None:
        assert init_rotation is None and init_translation is None, "pose_disturbance and init_rotation/init_translation cannot be set at the same time"
        new_pose = frame.get_disturbance(translation_disturbance=pose_disturbance[0], rotation_disturbance=pose_disturbance[1])
        init_rotation = matrix_to_quaternion(new_pose[:3, :3]).cuda()
        init_translation = new_pose[:3, 3].cuda()
    gaussians.eval_setup(op, opt, init_rotation, init_translation)
    
    viewpoint_cam = frame.get_camera(set_to_identity=True)
    gt_image = viewpoint_cam.original_image.cuda()
    # torchvision.utils.save_image(gt_image, save_dir / "gt.png")
    gaussians.optimizer.zero_grad(set_to_none = True)

    progress_bar = tqdm.tqdm(range(opt.eval_iterations), desc="optimize progress")
    
    for i in range(opt.eval_iterations):
        gaussians.assign_transform_from_delta_pose()
        lr = gaussians.update_learning_rate_for_delta_translation(i)
        print("iteration: {}, lr: {}".format(i, lr))

        pipe.convert_SHs_python = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim(image, gt_image))

        image_loss.backward()

        with torch.no_grad():

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if i == opt.eval_iterations - 1 or i % 10 == 0:
                # torchvision.utils.save_image(image, save_dir / f"{comment}_{i}_render.png")

                delta_rotation, delta_translation = gaussians.get_delta_rotation, gaussians.get_delta_translation
                delta_rotation = quaternion_to_matrix(delta_rotation)[0]
                pred_pose = torch.cat((delta_rotation, delta_translation.T), -1).cpu().numpy()
                evler.update(gt_pose, pred_pose)
                degree, cm = evler.get_current_degree_cm()
                    
            progress_bar.set_description(f"loss: {image_loss.item():.4f} degree: {degree}, cm: {cm}")
            progress_bar.update(1)
    _degree, _cm = evler.get_total_degree_and_cm()
    
    np.save(save_dir / f"{image_id}_{pose_disturbance[0]}_{pose_disturbance[1]}_{comment}_degree.npy", _degree)
    np.save(save_dir / f"{image_id}_{pose_disturbance[0]}_{pose_disturbance[1]}_{comment}_cm.npy", _cm)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    mp = MyParams(parser)
    pp = PipelineParams(parser)
    opt = OptimizierParams(parser)
    parser.add_argument('--load_iteration', type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    
    # init_pose = np.loadtxt("debug_track/167_pred.txt")
    # init_translation = torch.from_numpy(init_pose[:3, 3])
    # init_rotation = matrix_to_quaternion(torch.from_numpy(init_pose[:3, :3]))

    ## Onepose
    # X_directions = [0, 45]
    # Y_directions = np.arange(0, 360, 45)

    # for init_number, (x_direction, y_direction) in enumerate(product(X_directions, Y_directions)):
    #     init_rotation = R.from_euler("XYZ", [x_direction, y_direction, 180], degrees=True) # for z direction, we simply omit it for the symmetry
    #     init_rotation = torch.from_numpy(init_rotation.as_matrix()).float()
    #     init_translation = torch.tensor([0, 0, 0])
    #     init_rotation = matrix_to_quaternion(init_rotation)
    #     eval(0, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), init_translation, init_rotation, init_number)
    
    ## NeRF
    # X_directions = [135, 90]
    # Z_directions = np.arange(0, 360, 90)

    # for init_number, (x_direction, z_direction) in enumerate(product(X_directions, Z_directions)):
    #     init_rotation = R.from_euler("XYZ", [x_direction, 0, z_direction], degrees=True) # for z direction, we simply omit it for the symmetry
    #     init_rotation = torch.from_numpy(init_rotation.as_matrix()).float()
    #     init_translation = torch.tensor([0, 0, 0])
    #     init_rotation = matrix_to_quaternion(init_rotation)
    #     eval(0, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), init_translation, init_rotation, init_number)
    #     break

    # translation_range = [i / 100 for i in range(0, 26, 2)]
    # for item_number in [1, 20, 54, 105, 153]: ## random number
    for item_number in [1, 20, 54, 105, 153]:
        for i in range(5):
            eval(item_number, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), opt.extract(args), pose_disturbance=(0.15, 15), comment=i)

    