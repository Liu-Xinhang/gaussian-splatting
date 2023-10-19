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

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.pose_utils import matrix_to_quaternion, quaternion_to_matrix
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.optimizer_rate import *

class GaussianModel: ## 这个类存储的就是3d gaussian

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._new_xyz = torch.empty(0)
        self._new_features_dc = torch.empty(0)
        self._new_features_rest = torch.empty(0)
        self._new_scaling = torch.empty(0)
        self._new_rotation = torch.empty(0)
        self._new_opacity = torch.empty(0) 

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = None
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_new_scaling(self):
        return self.scaling_activation(self._new_scaling)
    
    @property
    def get_init_scaling(self):
        return self.scaling_activation(self._scaling_bak)
    
    @property ## 只读属性
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_init_rotation(self):
        return self.rotation_activation(self._rotation_bak)
    
    @property
    def get_new_rotation(self):
        return self.rotation_activation(self._new_rotation)
    
    @property
    def get_new_xyz(self):
        return self._new_xyz
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_init_xyz(self):
        return self._xyz_bak
    
    @property
    def get_origin_xyz(self):
        return self._xyz_origin
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_init_opacity(self):
        return self.opacity_activation(self._opacity_bak)

    @property
    def get_new_opacity(self):
        return self.opacity_activation(self._new_opacity)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_delta_rotation(self):
        assert hasattr(self, "_delta_rotation"), "only in eval mode can have delta_rotation"
        return self.rotation_activation(self._delta_rotation)
    
    @property
    def get_delta_translation(self):
        assert hasattr(self, "_delta_translation"), "only in eval mode can have delta_translation"
        return self._delta_translation

    def reset_transform(self):
        self._xyz = self._xyz_bak
        self._rotation = self._rotation_bak
    
    def assign_transform(self, rotation, translation, bundle=False):
        rotation_matrix = quaternion_to_matrix(self.get_init_rotation)
        self._rotation = matrix_to_quaternion(rotation @ rotation_matrix).detach()
        self._xyz = ((rotation @ self.get_init_xyz.T).T + translation).detach()
        if bundle:
            self._rotation_bak = self._rotation.detach().clone() 
            self._xyz_bak = self._xyz.detach().clone()
    
    def refresh_transform(self, use_inertia=False, bundle=False):
        if bundle:
            new_gaussian_number = self._new_opacity.shape[0]
            if new_gaussian_number == 0:
                new_gaussian_number = -self._rotation.shape[0]
            # self._rotation_bak = self._rotation[:-new_gaussian_number].detach().clone()
            # self._xyz_bak = self._xyz[:-new_gaussian_number].detach().clone()
            # self._scaling = self._scaling[:-new_gaussian_number]
            # self._features_dc = self._features_dc[:-new_gaussian_number]
            # self._features_rest = self._features_rest[:-new_gaussian_number]
            # self._opacity = self._opacity[:-new_gaussian_number]
            self._rotation_bak = self._rotation.detach().clone()[:-new_gaussian_number]
            self._xyz_bak = self._xyz.detach().clone()[:-new_gaussian_number]
            
            rotation_matrix = quaternion_to_matrix(self.get_new_rotation) # N, 3, 3
            delta_rotation_matrix = quaternion_to_matrix(self.get_delta_rotation) # 1, 3, 3
            _new_rotation = matrix_to_quaternion(delta_rotation_matrix @ rotation_matrix) # N, 4
            _new_xyz = (delta_rotation_matrix[0] @ self.get_new_xyz.T).T + self.get_delta_translation

            self._new_xyz = self.replace_tensor_to_optimizer(_new_xyz, "_new_xyz")["_new_xyz"]
            self._new_rotation = self.replace_tensor_to_optimizer(_new_rotation, "_new_rotation")["_new_rotation"]
            
        else:
            self._rotation_bak = self._rotation.detach().clone()
            self._xyz_bak = self._xyz.detach().clone()

        if not use_inertia:
            self._delta_rotation = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device="cuda", requires_grad=True)
            self._delta_translation = torch.zeros((1, 3), dtype=torch.float32, device="cuda", requires_grad=True)

    def assign_transform_from_delta_pose(self, bundle=False):
        if bundle:
            _rotation = torch.concat([self.get_init_rotation, self.get_new_rotation])
            _xyz = torch.concat([self._xyz_bak, self._new_xyz])
            self._scaling = torch.concat([self._scaling_bak, self._new_scaling])
            self._features_dc = torch.concat([self._features_dc_bak, self._new_features_dc])
            self._features_rest = torch.concat([self._features_rest_bak, self._new_features_rest])
            self._opacity = torch.concat([self._opacity_bak, self._new_opacity])
        else:
            _rotation = self.get_init_rotation
            _xyz = self.get_init_xyz
        rotation_matrix = quaternion_to_matrix(_rotation) # N, 3, 3
        delta_rotation_matrix = quaternion_to_matrix(self.get_delta_rotation) # 1, 3, 3
        self._rotation = matrix_to_quaternion(delta_rotation_matrix @ rotation_matrix) # N, 4
        self._xyz = (delta_rotation_matrix[0] @ _xyz.T).T + self.get_delta_translation
        # self._xyz.retain_grad()
        # self._rotation.retain_grad()

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [ ## 这个应该是Adam优化器可以识别的格式
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    def set_optimizer(self, eval_args, opt):
        l = [ ## 这个应该是Adam优化器可以识别的格式
            {'params': [self._delta_translation], 'lr': opt.delta_translation_init*self.spatial_lr_scale, "name": "delta_translation"},
            {'params': [self._delta_rotation], 'lr': eval_args.rotation_lr, "name": "delta_rotation"},
            
            {'params': [self._new_xyz], 'lr': eval_args.position_lr_init, "name": "_new_xyz"},
            {'params': [self._new_features_dc], 'lr': eval_args.feature_lr, "name": "_new_f_dc"},
            {'params': [self._new_features_rest], 'lr': eval_args.feature_lr / 20.0, "name": "_new_f_rest"},
            {'params': [self._new_opacity], 'lr': eval_args.opacity_lr, "name": "_new_opacity"},
            {'params': [self._new_scaling], 'lr': eval_args.scaling_lr, "name": "_new_scaling"},
            {'params': [self._new_rotation], 'lr': eval_args.rotation_lr, "name": "_new_rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        if opt.scheduler_type == "constant":
            self.delta_translation_scheduler_args = get_constant_lr_func(opt.delta_translation_init*self.spatial_lr_scale)
        elif opt.scheduler_type == "constant_then_expon_lr":
            self.delta_translation_scheduler_args = get_constant_then_expon_lr_func(lr_init=opt.delta_translation_init*self.spatial_lr_scale,
                                                    lr_final=opt.delta_translation_final*self.spatial_lr_scale,
                                                    const_step=opt.eval_iterations//2,
                                                    max_steps=opt.eval_iterations)
        elif opt.scheduler_type == "multi_step":
            self.delta_translation_scheduler_args = get_multi_step_lr_func(opt.delta_translation_init * self.spatial_lr_scale, [50, 100, 150], 0.8)
        else:
            raise NotImplementedError
        
    def eval_setup(self, eval_args, opt, rotation=None, translation=None):
        self.percent_dense = eval_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        self._new_xyz = torch.empty(0, *self._xyz.shape[1:], device="cuda")
        self._new_features_dc = torch.empty(0, *self._features_dc.shape[1:], device="cuda")
        self._new_features_rest = torch.empty(0, *self._features_rest.shape[1:], device="cuda")
        self._new_scaling = torch.empty(0, *self._scaling.shape[1:], device="cuda")
        self._new_rotation = torch.empty(0, *self._rotation.shape[1:], device="cuda")
        self._new_opacity = torch.empty(0, *self._opacity.shape[1:], device="cuda")

        self._features_dc_bak = self._features_dc.detach().clone()
        self._features_rest_bak = self._features_rest.detach().clone()
        self._scaling_bak = self._scaling.detach().clone()
        self._opacity_bak = self._opacity.detach().clone() 
        
        if rotation is None:
            self._delta_rotation = torch.tensor([[1, 0, 0, 0]], dtype=torch.float32, device="cuda", requires_grad=True)
        else:
            self._delta_rotation = rotation.reshape(1, 4).float().cuda().requires_grad_(True)
        if translation is None:
            self._delta_translation = torch.zeros((1, 3), dtype=torch.float32, device="cuda", requires_grad=True)
        else:
            self._delta_translation = translation.reshape(1, 3).float().cuda().requires_grad_(True)
        

        self.set_optimizer(eval_args, opt)
        
        self._fix_parameter()

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
    
    def update_learning_rate_for_delta_translation(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "delta_translation":
                lr = self.delta_translation_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def move_new_guassian_to_old(self, eval_args):
        ## 这里要detach因为我们的newxyz是有梯度的，但是这里我们希望从此他就不要有梯度了
        self._xyz_bak = torch.cat((self._xyz_bak, self._new_xyz), dim=0).detach()
        self._rotation_bak = torch.cat((self._rotation_bak, self._new_rotation), dim=0).detach()
        self._features_dc_bak = torch.cat((self._features_dc_bak, self._new_features_dc), dim=0).detach()
        self._features_rest_bak = torch.cat((self._features_rest_bak, self._new_features_rest), dim=0).detach()
        self._scaling_bak = torch.cat((self._scaling_bak, self._new_scaling), dim=0).detach()
        self._opacity_bak = torch.cat((self._opacity_bak, self._new_opacity), dim=0).detach()

        self.reset_new(eval_args)
        
    def reset_new(self, eval_args):
        self._new_xyz = torch.empty(0, *self._xyz.shape[1:], device="cuda")
        self._new_features_dc = torch.empty(0, *self._features_dc.shape[1:], device="cuda")
        self._new_features_rest = torch.empty(0, *self._features_rest.shape[1:], device="cuda")
        self._new_scaling = torch.empty(0, *self._scaling.shape[1:], device="cuda")
        self._new_rotation = torch.empty(0, *self._rotation.shape[1:], device="cuda")
        self._new_opacity = torch.empty(0, *self._opacity.shape[1:], device="cuda") 

        # self._new_xyz = self.replace_tensor_to_optimizer(new_xyz, "_new_xyz")["_new_xyz"]
        # self._new_rotation = self.replace_tensor_to_optimizer(new_rotation, "_new_rotation")["_new_rotation"]
        # self._new_features_dc = self.replace_tensor_to_optimizer(new_features_dc, "_new_f_dc")["_new_f_dc"]
        # self._new_features_rest = self.replace_tensor_to_optimizer(new_features_rest, "_new_f_rest")["_new_f_rest"]
        # self._new_scaling = self.replace_tensor_to_optimizer(new_scaling, "_new_scaling")["_new_scaling"]
        # self._new_opacity = self.replace_tensor_to_optimizer(new_opacity, "_new_opacity")["_new_opacity"]
        self.set_optimizer(eval_args)


    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_camera_extent(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
            self.spatial_lr_scale = float(lines[0].strip())
            print("camera extent: ", self.spatial_lr_scale)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self._xyz_bak = self._xyz.detach().clone()
        self._rotation_bak = self._rotation.detach().clone()

        self._xyz_origin = self._xyz.detach().clone() ## 供球谐函数

        self.active_sh_degree = self.max_sh_degree

    def _fix_delta_translation(self):
        self._delta_translation.requires_grad_(False)
        
    def _fix_parameter(self):
        self._features_dc.requires_grad_(False)
        self._features_rest.requires_grad_(False)
        self._opacity.requires_grad_(False)
        self._scaling.requires_grad_(False)
        # self._xyz.requires_grad_(False)
        # self._rotation.requires_grad_(False)
            
    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is None:
                    optimizable_tensors[group["name"]] = nn.Parameter(tensor.requires_grad_(True))
                    break
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor) ## step 保持不变

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ("delta_rotation", "delta_translation"):
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimzer_bundle(self, tensors_dict):
        optimizable_tensors = {}
        rows_to_concat = {}
        for group in self.optimizer.param_groups:
            if group["name"] in ("delta_translation", "delta_rotation"):
                continue
            extension_tensor = tensors_dict[group["name"].lstrip("_new_")]
            if extension_tensor.shape[0] == 0: ## 证明没有更新的点
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"].lstrip("_new_")] = group["params"][0]
                rows_to_concat[group["name"].lstrip("_new_")] = extension_tensor.shape[0]
            else:
                if group["params"][0].shape[0] == 0:
                    group["params"][0] = nn.Parameter(extension_tensor, requires_grad=True)
                else:
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"].lstrip("_new_")] = group["params"][0]
                rows_to_concat[group["name"].lstrip("_new_")] = extension_tensor.shape[0]

        return optimizable_tensors, rows_to_concat
    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, bundle):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}
        if bundle:
            optimizable_tensors, rows_to_concat = self.cat_tensors_to_optimzer_bundle(d)
            if "xyz" in optimizable_tensors:    
                self._new_xyz = optimizable_tensors["xyz"]
                self._new_features_dc = optimizable_tensors["f_dc"]
                self._new_features_rest = optimizable_tensors["f_rest"]
                self._new_opacity = optimizable_tensors["opacity"]
                self._new_scaling = optimizable_tensors["scaling"]
                self._new_rotation = optimizable_tensors["rotation"]
            shape = self._opacity_bak.shape[0] + self._new_xyz.shape[0]
            self.xyz_gradient_accum = torch.zeros((shape, 1), device="cuda")
            self.denom = torch.zeros((shape, 1), device="cuda")
            self.max_radii2D = torch.zeros((shape), device="cuda")
        else:
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = torch.zeros((self.get_opacity.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_opacity.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_opacity.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, bundle=False):
        n_init_points = self.get_opacity.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        if bundle:
            rots = build_rotation(self._rotation_bak[selected_pts_mask]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_init_xyz[selected_pts_mask].repeat(N, 1)
            new_rotation = self._rotation_bak[selected_pts_mask].repeat(N, 1)
        else:
            rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
            new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N)) ## 这里的0.8是一个经验值
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, bundle)
        if not bundle:
            prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.prune_points(prune_filter) ## 这里是根据grad和区域大小来进行prune
        else:
            valid_points_mask = ~torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]

            ## 分别更新，提取optimizer中新的点
            new_gaussian_number = self._new_xyz.shape[0]
            if new_gaussian_number == 0: ## 保证-new_gaussian_points是有效的表达
                return
            
            ## 因此我们可以直接对原始的xyz进行更新
            selected_pts_mask_front = ~selected_pts_mask[:grads.shape[0]]
            assert selected_pts_mask[grads.shape[0]:].sum() == 0
            self._xyz_bak = self._xyz_bak[:-new_gaussian_number][selected_pts_mask_front]
            self._rotation_bak = self._rotation_bak[:-new_gaussian_number][selected_pts_mask_front]
            self._scaling = self._scaling[:-new_gaussian_number][selected_pts_mask_front]
            self._features_dc = self._features_dc[:-new_gaussian_number][selected_pts_mask_front]
            self._features_rest = self._features_rest[:-new_gaussian_number][selected_pts_mask_front]
            self._opacity = self._opacity[:-new_gaussian_number][selected_pts_mask_front]
            
            ## 然后我们concat，因为prune_filter中没有内容需要prune掉（因为全部是torch.zeros)，所以这里我们直接concat
            self._xyz_bak = torch.concat([self._xyz_bak, self._new_xyz])
            self._rotation_bak = torch.concat([self._rotation_bak, self._new_rotation])
            self._features_dc = torch.concat([self._features_dc, self._new_features_dc])
            self._features_rest = torch.concat([self._features_rest, self._new_features_rest])
            self._opacity = torch.concat([self._opacity, self._new_opacity])
            self._scaling = torch.concat([self._scaling, self._new_scaling])


    def densify_and_clone(self, grads, grad_threshold, scene_extent, bundle):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if bundle:
            new_xyz = self._xyz_bak[selected_pts_mask]
            new_rotation = self._rotation_bak[selected_pts_mask]
            new_features_dc = self._features_dc_bak[selected_pts_mask]
            new_features_rest = self._features_rest_bak[selected_pts_mask]
            new_opacities = self._opacity_bak[selected_pts_mask]
            new_scaling = self._scaling_bak[selected_pts_mask]
        else:
            new_xyz = self._xyz[selected_pts_mask]
            new_rotation = self._rotation[selected_pts_mask]
            new_features_dc = self._features_dc[selected_pts_mask]
            new_features_rest = self._features_rest[selected_pts_mask]
            new_opacities = self._opacity[selected_pts_mask]
            new_scaling = self._scaling[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, bundle)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, bundle=False):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, bundle=bundle) ## 注意必须先clone再split
        if not bundle:
            self.densify_and_split(grads, max_grad, extent, bundle=bundle) ## bundle的时候暂时不要delete点了
        
        if bundle:
            _opacity = torch.cat((self.get_init_opacity, self.get_new_opacity))
            prune_mask = (_opacity < min_opacity).squeeze()
        else:
            prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            if bundle:
                _scaling = torch.cat((self.get_init_scaling, self.get_new_scaling))
                big_points_ws = _scaling.max(dim=1).values > 0.1 * extent
            else:
                big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if not bundle:
            self.prune_points(prune_mask) ## 这里是根据opacity以及区域大小来prune
        else:
            ## 更新的原理就是我们新增加的量都在最后
            valid_points_mask = ~prune_mask
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

            self.denom = self.denom[valid_points_mask]
            self.max_radii2D = self.max_radii2D[valid_points_mask]

            new_gaussian_number = self._new_features_dc.shape[0]
            if new_gaussian_number == 0:
                new_gaussian_number = -prune_mask.shape[0]
            
            prune_mask_origin = valid_points_mask[:-new_gaussian_number]
            prune_mask_new = valid_points_mask[-new_gaussian_number:]
            
            self._xyz_bak = self._xyz_bak[prune_mask_origin]
            self._rotation_bak = self._rotation_bak[prune_mask_origin]
            self._features_dc_bak = self._features_dc_bak[prune_mask_origin]
            self._features_rest_bak = self._features_rest_bak[prune_mask_origin]
            self._opacity_bak = self._opacity_bak[prune_mask_origin]
            self._scaling_bak = self._scaling_bak[prune_mask_origin]

            optimizable_tensors = self._prune_optimizer(prune_mask_new)

            self._new_xyz = optimizable_tensors["_new_xyz"]
            self._new_features_dc = optimizable_tensors["_new_f_dc"]
            self._new_features_rest = optimizable_tensors["_new_f_rest"]
            self._new_opacity = optimizable_tensors["_new_opacity"]
            self._new_scaling = optimizable_tensors["_new_scaling"]
            self._new_rotation = optimizable_tensors["_new_rotation"]

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1