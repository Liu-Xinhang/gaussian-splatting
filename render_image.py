from argparse import ArgumentParser, Namespace
from scene import GaussianModel
from scene.frame import Frame
import sys
from arguments import ModelParams, OptimizationParams, MyParams, PipelineParams
from gaussian_renderer import render
import torch
import torchvision
from pathlib import Path
import tqdm

def render_image(image_id, dataset, opt, pipe, load_iteration, myparms):
    gaussians = GaussianModel(dataset.sh_degree)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    frame = Frame(image_id, dataset, gaussians, load_iteration, cameras_extent=0.5027918756008148, myparms=myparms)

    viewpoint_cam = frame.get_camera(set_to_identity=False) ## 先渲染一个把位姿作用在相机视角上的版本
    Frame.gaussians.reset_transform()
    render_pkg = render(viewpoint_cam, Frame.gaussians, pipe, background)
    image_ = render_pkg["render"]

    viewpoint_cam = frame.transform(1)

    render_pkg = render(viewpoint_cam, Frame.gaussians, pipe, background)

    image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
    gt_image = viewpoint_cam.original_image

    save_dir = Path("debug")
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
    
    for image_id in tqdm.trange(737):
        render_image(image_id, lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args))

    