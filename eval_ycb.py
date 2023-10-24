from argparse import ArgumentParser, Namespace
from scene import GaussianModel
from scene.frame import YCBVFrame
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
from utils.loss_utils import l1_loss, ssim, mask_loss, dice_loss, iou_loss, weighted_mse_loss, Optical_loss
from utils.pose_utils import matrix_to_quaternion, euler_angles_to_matrix, quaternion_to_matrix
import os
import json
from utils.system_utils import searchForMaxIteration
from utils.geometry_utils import load_ply, calculate_models_diameter
from utils.eval_utils import DegreeAndCM, AddsAndShift
from utils.image_utils import PutTheSecondImageOnTheFirstImageWithOpacity

def eval(image_id, dataset, op, pipe, load_iteration, myparms, opt, model_points, diameter, K, sequence_num):
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

    if myparms.mytype=="YCBV":
        frame = YCBVFrame(image_id, dataset, gaussians)
    else:
        raise NotImplementedError
    seq_num = Path(dataset.source_path).stem
    obj_name = Path(dataset.model_path).parent.stem
    save_dir = Path(f"debug_YCBV_optical/debug_data_{obj_name}_{seq_num}")
    save_dir.mkdir(exist_ok=True, parents=True)

    evler = DegreeAndCM(translation_scale="m")

    evler_add = AddsAndShift(model_points, diameter, torch.device("cuda:0"), K)

    rotation, translation = frame.get_rotation_translation()
    gt_pose = torch.cat((rotation, translation[:, None]), -1).cpu().numpy()
    print("gt_pose:", gt_pose)


    ## init pose
    new_pose = frame.get_disturbance()
    print("init_pose:", new_pose)
    init_rotation = matrix_to_quaternion(new_pose[:3, :3]).cuda()
    init_translation = new_pose[:3, 3].cuda()
    evler.update(gt_pose, new_pose.cpu().numpy())
    evler_add.add_new_frame(gt_pose, new_pose.cpu().numpy())
    
    degree, cm = evler.get_current_degree_cm()
    add, adds, _ = evler_add.get_current_add_adds_shift()
    print(f"init: degree: {degree:.2f}, cm: {cm:.2f} add: {add:.2f}, adds: {adds:.2f}")

    gaussians.eval_setup(op, opt, init_rotation, init_translation)
    
    viewpoint_cam = frame.get_camera(set_to_identity=True)
    gt_image = viewpoint_cam.original_image.cuda() # 3, H, W
    gt_mask = viewpoint_cam.original_mask.cuda() # H, W
    gt_mask_vis = viewpoint_cam.original_mask_vis.cuda() # H, W
    if myparms.save_image:
        torchvision.utils.save_image(gt_image, save_dir / f"{frame.image_name:04d}_gt.png")
        torchvision.utils.save_image(gt_mask_vis, save_dir / f"{frame.image_name:04d}_gt_mask_vis.png")
        torchvision.utils.save_image(gt_mask, save_dir / f"{frame.image_name:04d}_gt_mask.png")
    gaussians.optimizer.zero_grad(set_to_none = True)

    progress_bar = tqdm.tqdm(range(opt.eval_iterations), desc="optimize progress")
    optical_loss = Optical_loss()
    
    for i in range(opt.eval_iterations):
        gaussians.assign_transform_from_delta_pose()
        lr = gaussians.update_learning_rate_for_delta_translation(i)
        print("iteration: {}, lr: {}".format(i, lr))

        pipe.convert_SHs_python = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        ## compute mask
        if i <= opt.eval_iterations // 4:
            render_pkg = render(viewpoint_cam, gaussians, pipe, torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda") if not dataset.white_background else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))
            image_w, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            mask_pred = 1 - (image_w - image).mean(0)
            image_loss = iou_loss(mask_pred , gt_mask)
            if myparms.save_image:
                torchvision.utils.save_image(mask_pred, save_dir / f"{frame.image_name:04d}_{i}_mask.png")

        else:
            gaussians._fix_delta_translation()
            # Ll1 = l1_loss(image * gt_mask_vis, gt_image * gt_mask_vis)
            # image_loss = (1.0 - op.lambda_dssim) * Ll1 + op.lambda_dssim * (1.0 - ssim(image * gt_mask_vis, gt_image * gt_mask_vis))
            # image_loss = weighted_mse_loss(image * gt_mask_vis, gt_image * gt_mask_vis)
            image_loss = optical_loss(image * gt_mask_vis, gt_image * gt_mask_vis, gt_mask_vis, save_dir / f"{frame.image_name:04d}_{i}_optical.png", myparms.save_image)

        image_loss.backward()

        with torch.no_grad():

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)
            if i == opt.eval_iterations - 1 or i % 10 == 0:
                if myparms.save_image:
                    torchvision.utils.save_image(image * gt_mask_vis, save_dir / f"{frame.image_name:04d}_{i}_render.png")
                    image = PutTheSecondImageOnTheFirstImageWithOpacity(gt_image.cpu().numpy(), (image * gt_mask_vis).cpu().numpy(), 128)
                    image.save(save_dir / f"{frame.image_name:04d}_{i}_combine.png")

                delta_rotation, delta_translation = gaussians.get_delta_rotation, gaussians.get_delta_translation
                delta_rotation = quaternion_to_matrix(delta_rotation)[0]
                pred_pose = torch.cat((delta_rotation, delta_translation.T), -1).cpu().numpy()
                evler.update(gt_pose, pred_pose)
                evler_add.add_new_frame(gt_pose, pred_pose)
                degree, cm = evler.get_current_degree_cm()
                add, adds, _ = evler_add.get_current_add_adds_shift()
                print(gt_pose, pred_pose)
                    
            progress_bar.set_description(f"loss: {image_loss.item():.4f} degree: {degree:.5f}, cm: {cm:.5f}, add: {add:.2f}, adds: {adds:.2f}")
            progress_bar.update(1)
    _degree, _cm = evler.get_total_degree_and_cm()
    _add, _adds = evler_add.get_total_add_and_adds()
    
    np.save(save_dir / f"{frame.image_name:04d}_degree.npy", _degree)
    np.save(save_dir / f"{frame.image_name:04d}_cm.npy", _cm)
    np.save(save_dir / f"{frame.image_name:04d}_add.npy", _add)
    np.save(save_dir / f"{frame.image_name:04d}_adds.npy", _adds)
    np.save(save_dir / f"{frame.image_name:04d}_pred_pose.npy", pred_pose)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    mp = MyParams(parser)
    pp = PipelineParams(parser)
    opt = OptimizierParams(parser)
    parser.add_argument('--load_iteration', type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    datasets = lp.extract(args)
    
    if "_" in Path(datasets.model_path).parent.stem:
        obj_number = int(Path(datasets.model_path).parent.stem.split("_")[0])
    else:
        obj_number = int(Path(datasets.model_path).parent.stem)
    model_points = load_ply(Path(datasets.source_path).parent.parent / "models_eval" / f"obj_{obj_number:06d}.ply") / 1000
    ## read model diameter
    with open("temp_datasets/ycbv/models_eval/models_info.json") as file:
        model_info = json.load(file)
    diameter = model_info[str(obj_number)]["diameter"] / 1000
    print("diameter(m): ", diameter)

    ## load K
    ## choose the first image K
    with open(Path(datasets.source_path) / "scene_camera.json") as file:
        K = np.array(json.load(file)["1"]["cam_K"]).reshape(3, 3)
    
    ## get total number of images
    sequence_number = Path(datasets.source_path).stem
    with open("temp_datasets/ycbv/image_lists/test.txt") as file: #! hard code
        lines = file.readlines()
    total_length = len([line.strip() for line in lines if line.startswith(sequence_number)])
    # for item_number in range(total_length):
    for item_number in [0]:
        print(f"process {item_number}/{total_length}")
        eval(item_number, datasets, op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), opt.extract(args), model_points, diameter, K, sequence_number)

    