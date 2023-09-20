from argparse import ArgumentParser, Namespace
from scene import GaussianModel
from scene.frame import Frame
import sys
from arguments import ModelParams, OptimizationParams, MyParams, PipelineParams
from gaussian_renderer import render
import torch
import torchvision
from pathlib import Path
import numpy as np
import tqdm
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import matrix_to_quaternion
import os
from utils.system_utils import searchForMaxIteration

def eval(image_id, dataset, opt, pipe, load_iteration, myparms, init_translation, init_rotation):
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
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    frame = Frame(image_id, dataset, gaussians, load_iteration, cameras_extent=0.5027918756008148, myparms=myparms)

    save_dir = Path("debug_render")
    save_dir.mkdir(exist_ok=True)

    rotation, translation = frame.get_rotation_translation
    rotation = matrix_to_quaternion(rotation)
    Frame.gaussians.eval_setup(opt, rotation, init_translation)
    
    viewpoint_cam = frame.get_camera(set_to_identity=True)
    gt_image = viewpoint_cam.original_image.cuda()
    torchvision.utils.save_image(gt_image, save_dir / "gt.png")
    Frame.gaussians.optimizer.zero_grad(set_to_none = True)

    progress_bar = tqdm.tqdm(range(1000), desc="optimize progress")
    
    for i in range(1000):
        Frame.gaussians.assign_transform_from_delta_pose()
        render_pkg = render(viewpoint_cam, Frame.gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    
        Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        image_loss.backward()

        with torch.no_grad():
            progress_bar.set_description(f"loss: {image_loss.item():.4f}")
            progress_bar.update(1)

            Frame.gaussians.optimizer.step()
            Frame.gaussians.optimizer.zero_grad(set_to_none = True)
            if i == 999 or i % 10 == 0:
                torchvision.utils.save_image(image, save_dir / f"{i}_render.png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    mp = MyParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    

    init_pose = np.loadtxt("temp_datasets/loquat-2/poses_ba/100.txt")
    init_translation = torch.from_numpy(init_pose[:3, 3])
    init_rotation = matrix_to_quaternion(torch.from_numpy(init_pose[:3, :3]))

    eval(0, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), init_translation, init_rotation)

    