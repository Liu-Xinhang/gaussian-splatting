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
import time
from utils.loss_utils import l1_loss, ssim
from utils.pose_utils import quaternion_to_matrix
import os
from utils.system_utils import searchForMaxIteration
from utils.eval_utils import DegreeAndCM

## 暂时设置一个全局的变量
BUNDLE_ADJUSTMENT = False

def eval(dataset, opt, pipe, load_iteration, myparms, init_translation, init_rotation):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.spatial_lr_scale = 0.5027918756008148
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

    gaussians.eval_setup(opt)
    gaussians.assign_transform(init_rotation, init_translation)

    # scale = gaussians.get_scaling
    # np.save("scale.npy", scale.detach().cpu().numpy())
    # sys.exit()
    
    save_dir = Path("debug_track")
    save_dir.mkdir(exist_ok=True)
    
    ## glob all images
    image_root_dir = Path(dataset.source_path) / "input"
    total_images_path = image_root_dir.glob("*.png")
    total_images_path = sorted(total_images_path, key=lambda x: int(x.stem))

    ## evler
    evler = DegreeAndCM(translation_scale="m")

    iter_start = torch.cuda.Event(enable_timing = True) ## 记录gpu时间
    iter_end = torch.cuda.Event(enable_timing = True) ## 记录gpu时间
    
    for image_id, image_path in enumerate(total_images_path):
        start_time = time.time()
        print(f"eval {str(image_path)}")
        gaussians.refresh_transform(use_inertia=True)
        gaussians.set_optimizer(opt)    

        frame = Frame(image_id, dataset, gaussians, load_iteration, cameras_extent=0.5027918756008148, myparms=myparms)

        gt_rotation, gt_translation = frame.get_rotation_translation 
        gt_pose = torch.concat((gt_rotation, gt_translation.unsqueeze(-1)), -1).cpu().numpy()
        
        viewpoint_cam = frame.get_camera(set_to_identity=True)
        gt_image = viewpoint_cam.original_image.cuda()
        # torchvision.utils.save_image(gt_image, save_dir / f"{image_id}_gt.png")
        
        gaussians.optimizer.zero_grad(set_to_none = True)

        progress_bar = tqdm.tqdm(range(myparms.track_render_iterations), desc="optimize progress")

        for i in range(myparms.track_render_iterations):
            gaussians.assign_transform_from_delta_pose()
            pipe.convert_SHs_python = True ## 颜色根据原始的方向计算好球谐函数中的内容
            render_pkg = render(viewpoint_cam, Frame.gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
            Ll1 = l1_loss(image, gt_image)
            image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

            image_loss.backward()

            with torch.no_grad():
                progress_bar.set_description(f"loss: {image_loss.item():.4f}")
                progress_bar.update(1)

                if BUNDLE_ADJUSTMENT and i % 2 == 1: ##! 设置啥会儿去优化
                    size_threshold = 20
                    ## 我们会根据梯度，不透明度，camera的extent以及size的大小来决定保留哪些内容
                    # gaussians.densify_and_prune(0.0001, 0.005, 0.5027918756008148, size_threshold, bundle=BUNDLE_ADJUSTMENT) ##!
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 0.5027918756008148, size_threshold, bundle=BUNDLE_ADJUSTMENT)
                
                ## 优化器
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if i ==  myparms.track_render_iterations-1:
                    ## 每一次迭代的最后，我们再将记录grad，为bundle adjestment做准备
                    if BUNDLE_ADJUSTMENT:
                        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    # torchvision.utils.save_image(image, save_dir / f"{image_id}_render.png")
                    
                    ## eval pose accuracy
                    delta_rotation, delta_translation = gaussians.get_delta_rotation, gaussians.get_delta_translation
                    delta_rotation = quaternion_to_matrix(delta_rotation)[0]
                    pred_rotation = delta_rotation @ init_rotation
                    pred_translation = (delta_rotation @ init_translation.T).T + delta_translation
                    pred_pose = torch.concat([pred_rotation, pred_translation.T], -1).cpu().numpy()
                    init_rotation = pred_rotation
                    init_translation = pred_translation
                    evler.update(gt_pose, pred_pose)
                    degree, cm = evler.get_current_degree_cm()
                    
                    progress_bar.set_description("degree: {}, cm: {}".format(degree, cm))
                    end_time = time.time()
                    print(f"cost time: {end_time - start_time}")

    degree, cm = evler.get_total_degree_and_cm()
    np.save(save_dir / f"{myparms.track_render_iterations}_degree.npy", degree)
    np.save(save_dir / f"{myparms.track_render_iterations}_cm.npy", cm)
        

if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    mp = MyParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--load_iteration', type=int, default=None)
    args = parser.parse_args(sys.argv[1:])
    

    init_pose = np.loadtxt("temp_datasets/loquat-2/poses_ba/0.txt")
    init_translation = torch.from_numpy(init_pose[:3, 3:]).T.float().cuda()
    init_rotation = torch.from_numpy(init_pose[:3, :3]).float().cuda()

    eval(lp.extract(args), op.extract(args), pp.extract(args), args.load_iteration, mp.extract(args), init_translation, init_rotation)


    