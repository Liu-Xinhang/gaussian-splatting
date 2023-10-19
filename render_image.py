from argparse import ArgumentParser, Namespace
from scene import GaussianModel
from scene.frame import OneposeFrame, NeRFFrame, YCBVFrame
import sys
from arguments import ModelParams, OptimizationParams, MyParams, PipelineParams
from gaussian_renderer import render
import torch
import torchvision
from pathlib import Path
import tqdm
import os
from utils.system_utils import searchForMaxIteration

def render_image(image_id, dataset, opt, pipe, load_iteration, myparms):
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

    if myparms.mytype=="Onepose":
        frame = OneposeFrame(image_id, dataset, gaussians, load_iteration, cameras_extent=0.5027918756008148, myparms=myparms)
    elif myparms.mytype=="Nerf":
        frame = NeRFFrame(image_id, dataset, gaussians, myparms=myparms)
    elif myparms.mytype=="YCBV":
        frame = YCBVFrame(image_id, dataset, gaussians)
    else:
        raise NotImplementedError

    viewpoint_cam = frame.get_camera(set_to_identity=False) ## 先渲染一个把位姿作用在相机视角上的版本
    gaussians.reset_transform()
    pipe.convert_SHs_python = True
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    image_ = render_pkg["render"]

    viewpoint_cam = frame.transform(1)
    pipe.convert_SHs_python = True
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)

    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    gt_image = viewpoint_cam.original_image

    save_dir = Path("debug_YCBV")
    save_dir.mkdir(exist_ok=True)

    torchvision.utils.save_image(image, save_dir / f"{image_id}_render.png")
    torchvision.utils.save_image(gt_image, save_dir / f"{image_id}_gt.png")
    torchvision.utils.save_image(image_, save_dir / f"{image_id}_render_direct.png")

    
if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    mp = MyParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=int, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    args = parser.parse_args(sys.argv[1:])
    
    for image_id in tqdm.trange(1):
        render_image(image_id, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args))

    