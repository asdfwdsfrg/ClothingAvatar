from threading import local

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lib.config import cfg
from lib.networks.body_model import BodyModel
from lib.networks.embedder import *
from lib.utils.write_ply import write_ply
from torch.utils.cpp_extension import load

from . import embedder
torch.set_printoptions(threshold=np.inf)

class Network(nn.Module):
    def __init__(self):

        super(Network, self).__init__()

        cuda_module = load(name="blend_feature",
                    sources=["cuda_module/kernel/blend_feature.cpp", "cuda_module/kernel/blend_feature.cu"],
                    verbose=True)

        #encoder
        self.ec_ly1 = nn.Conv1d((72 + time_dim) * 128, 64 * 128, kernel_size= 1, groups =128)
        self.ec_ly2 = nn.Conv1d(64 * 128, 64 * 128, kernel_size= 1, groups =128)

        self.ec_ly21= nn.Conv1d(64 * 128, 8 * 128, kernel_size=1, groups=128)
        self.ec_ly22= nn.Conv1d(64 * 128, 8 * 128, kernel_size=1, groups=128)
       
        
        
        #decoder
        self.dc_ly1 = nn.Conv1d(80 * 128, 64 * 128, kernel_size= 1, groups =128)
        self.dc_ly21= nn.Conv1d(64 * 128, 32 * 128, kernel_size=1, groups=128)
        self.dc_ly22= nn.Conv1d(64 * 128, 3 * 128, kernel_size=1, groups=128)
       
        self.actvn = nn.ReLU()
        self.actvn2 = nn.Sigmoid()  
        # nodes x s' x (32+xyz_dim)

        #feature field
        self.f_ly1 = nn.Conv1d((32 + xyz_dim) * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly2 = nn.Conv1d(64 * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly3 = nn.Conv1d(64 * 128, 64 * 128, kernel_size=1, groups = 128)
        self.f_ly4 = nn.Conv1d(64 * 128, 256 * 128, kernel_size=1, groups = 128)

        #Nerf color
        self.c_ly1 = nn.Linear(256 + view_dim, 64)
        self.c_ly2 = nn.Linear(64, 3)

        #NerF density
        self.d_ly1 = nn.Linear(256, 1)
        self.cuda_module = load(name="blend_feature",
                  sources=["cuda_module/kernel/blend_feature.cpp", "cuda_module/kernel/blend_feature.cu"],
                   verbose=True)

    def encode(self, t_ped, poses, w):
        '''
        poses: batch x nodes_n x 24 x 3
        t_ped: batch x time_dim
        mean, std:  Batchs x nodes x 8
        w: attention_map : 128 x 24
        '''
        batch_size = poses.shape[0]
        nodes_n = cfg.n_nodes
        #B x N_nodes x 72
        poses = torch.mul(w, poses).view(batch_size,nodes_n,72)
        #B x N_nodes x time_dim
        t_ped = t_ped.expand(nodes_n, batch_size, time_dim).permute(1, 0, 2)
        #B x N_nodes X (72 + time_dim)
        encoder_in = torch.cat([poses, t_ped], dim = -1).permute(1, 2, 0).reshape(nodes_n * (72 + time_dim), batch_size)
        #nodes x 64 x batchs
        net = self.actvn(self.ec_ly1(encoder_in))
        net = self.actvn(self.ec_ly2(net))
        #nodes x 8 x batchs
        mean = self.ec_ly21(net).view(nodes_n, 8, batch_size).permute(0,2,1)
        logvar = self.ec_ly22(net).view(nodes_n, 8, batch_size).permute(0, 2, 1)
        return mean, logvar
    
    def decode(self, z, poses, w):
        '''
        poses: Batchs x 1 x 72
        z: nodes x Batchs x 8
        ei: Batchs x nodes x 32
        '''
        batch_size = poses.shape[0]
        nodes_n = cfg.n_nodes
        #B x N_nodes x 72
        poses = torch.mul(w, poses).view(batch_size,nodes_n,72)
        #N_nodes x 80 x batchs
        input = torch.cat([poses, z.transpose(0, 1)], dim = -1).permute(1, 2, 0).reshape(nodes_n * 80, batch_size)
        #N_nodes x 64 x batchs
        net = self.actvn(self.dc_ly1(input))
        ei = self.actvn(self.dc_ly21(net)).view(nodes_n, 32, batch_size).permute(0, 2, 1)
        delta_ni = self.actvn(self.dc_ly22(net)).view(nodes_n, 3, batch_size).permute(0, 2, 1)
        #nodes x batches x (32, 3)
        return ei, delta_ni

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mean)
        else:
            return mean
    
    def Feature_field(self, ei, local_coords, s_max):
        '''
        local_coords: nodes x B x s_max x 3
        ei:nodes x B x 32
        s: nodes x B 
        return 
            f:nodes x batch_size x s_max x 256
        '''
        nodes_n = ei.shape[0]
        batch_size = ei.shape[1] 
        ei = ei.expand(s_max, nodes_n,batch_size, ei.shape[-1])
        ei = ei.permute(1, 2, 0, 3)
        #N x B x s_max x (32 + xyz_dim)
        input = torch.cat([ei, local_coords], dim = -1).view(nodes_n,batch_size * s_max,(32 + xyz_dim)).permute(0, 2, 1)
        input = input.reshape(nodes_n * (32 + xyz_dim), batch_size * s_max)
        # nodes x (32 + xyz_dim) x (B * s_max)
        net = self.actvn(self.f_ly1(input))
        net = self.actvn(self.f_ly2(net))
        net = self.actvn(self.f_ly3(net))
        output = self.actvn(self.f_ly4(net))
        f = output.view(nodes_n, 256, batch_size, s_max).permute(0, 2, 3, 1)
        return f

    # def blend_feature(self, bweights, f):
    #     '''
    #     bweights: nodes x B x V
    #     f: nodes x B x V x 256
    #     '''
    #     bweights = bweights.permute(1, 2, 0)
    #     bweights += 0.0000001
    #     weights_sum = torch.sum(bweights, dim = -1)
    #     bweights = bweights / weights_sum[...,None]
    #     f = f.permute(1, 2, 3, 0)
    #     #B x V x 256
    #     f_blend = torch.matmul(f, bweights[...,None]).squeeze(dim = -1)
    #     return f_blend 

    # def blend_feature(self, bweights, f_hit, pts_ind):
    # #     '''
    # #     bweights: nodes x B x s_max x 1
    # #     f_hit: nodes x B x s_max x 256
    # #     pts_ind: nodes X B X s_max  indicates the pts_index 
    # #       f_blend: B x V x 256
    # #     '''
    #     f_blend = torch.zeros()
    #     f_ = f_hit * bweights
        




    def Nerf(self, f, viewdir):
        '''
        f: B x V x 256
        viewdir: B x n_rand x view_dim
        '''
        n_samples = cfg.N_samples
        rays = round(f.shape[1] / n_samples)
        viewdir = viewdir.expand(n_samples, f.shape[0],rays, view_dim).permute(1, 2, 0, 3).reshape(f.shape[0], f.shape[1], view_dim)
        input = torch.cat([f, viewdir], dim = -1) 
        netc = self.actvn(self.c_ly1(input))
        c = torch.sigmoid(self.c_ly2(netc))
        d = self.actvn(self.d_ly1(f))
        return c, d

    def wpts_to_smpl_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def world_dirs_to_pose_dirs(self, wdirs, Rh):
        """
        wdirs: n_batch, n_points, 3
        Rh: n_batch, 3, 3
        """
        pts = torch.matmul(wdirs, Rh)
        return pts

    def blend_weights(self, wpts, nodes_posed):
        """feature blending weights"""    
        '''
            wpts:n x B x V x 3
            nodes_posed: B x nodes_n x 3
            return:
                bweights: nodes_n X B X V  the nodes influence on each vertex
                mask: nodes x B X V  bool  knb[i][j] = 1 means Vj inside the nodes_i's influence range
        '''
        bweights = torch.zeros([nodes_posed.shape[1], wpts.shape[0], wpts.shape[1]], device = wpts.device)
        nodes_posed = nodes_posed.permute(1, 0, 2)
        norm_2 = torch.sum(torch.pow(wpts - nodes_posed[..., None, :], 2), dim = -1)
        nodes_influ = torch.exp(- norm_2 / (2 * torch.pow(torch.tensor(cfg.sigma), 2))) - torch.tensor(cfg.epsilon)
        mask = nodes_influ.ge(0)
        # hit_ind = torch.argwhere(mask == True)
        bweights = torch.max(nodes_influ, torch.tensor(0).float())
        return bweights, mask 

    def calculate_local_coords(self, wpts, nodes_T, nodes_weights, J, body, poses, shapes, R, Th):
        """get local_coords"""
        '''
            wpts: nodes_n x B x V x 4 
            nodes: B x nodes_n x 3
            nodes_delta: nodes_n x B x 3
            nodes_weights: nodes_n x Joints
            return:
                local_coords: nodes_n x B x V x 4
        '''
        #V x B x nodes_n x 4
        nodes_n = cfg.n_nodes
        wpts = wpts.permute(2,1,0,3)
        coords_T, j_transformed = body.get_lbs(poses, shapes, nodes_weights, wpts[...,:3], joints = J, inverse = True)
        #V x B x nodes_n x 3
        local_coords = coords_T - nodes_T 
        return local_coords.permute(2, 1, 0, 3)

    def forward(self, input, wpts, viewdir, body):
        #transform sampled points from world coord to smpl coord
        nodes_n = cfg.n_nodes
        batch_size =  input['R'].shape[0]
        #B x V x 3
        R = input['R']
        Th = input['Th']
        params = input['params']
        shapes = params['shapes']
        poses = params['poses'].squeeze(dim = -2)
        weights = body.basis['weights']
        nodes_ind = body.basis['nodes_ind']
        nodes_T = body.basis['v_shaped'][nodes_ind]
        nodes_weights= weights[nodes_ind]
        wpts = wpts.view(batch_size, -1, 3)
        pts_num = wpts.shape[1]
        ppts = self.wpts_to_smpl_pts(wpts, input)
        ppts = ppts.expand(nodes_n, batch_size, pts_num, 3)
        wviewdir = self.world_dirs_to_pose_dirs(viewdir, R)
        poses_exp = poses.expand(nodes_n, batch_size, 72).permute(1, 0, 2).reshape(batch_size, nodes_n, 24, 3)
        w = torch.matmul(body.attention_map, nodes_weights[..., None]).squeeze(dim=-1)
        w = w.expand(batch_size, 3, 128, 24).permute(0, 2, 3, 1)
        t_ped = time_embedder(input['latent_index']).view(batch_size, -1)
        #nodes in T pose
        batch_nodes_T = nodes_T.expand(batch_size, nodes_T.shape[-2], nodes_T.shape[-1])
        mean, logvar = self.encode(t_ped, poses_exp, w)
        z = self.reparameterize(mean, logvar)
        #nodes x B x (32, 3)
        ei, nodes_delta = self.decode(z, poses_exp, w)  
        batch_nodes_T = batch_nodes_T + nodes_delta.transpose(0, 1) 
        batch_nodes_posed, j_transformed = body.get_lbs(poses, shapes, nodes_weights, batch_nodes_T)
        bweights, mask = self.blend_weights(wpts, batch_nodes_posed)
        pts_ind = torch.arange(0, pts_num, device = wpts.device).expand(nodes_n, batch_size, pts_num).unsqueeze(dim = -1)
        pts_in = torch.cat([ppts, pts_ind], dim = -1)
        s_max = torch.max(torch.sum(mask, dim = -1))
        pts_hit = torch.zeros(nodes_n, batch_size, s_max, 4, device = wpts.device)
        self.cuda_module.launch_pts_hit(pts_in, mask, pts_hit, nodes_n, batch_size, pts_num, s_max) 
        torch.cuda.synchronize(wpts.device)
        #n x b x smax 
        pts_index = pts_hit[...,-1].to(torch.int64)
        bweights_ = torch.gather(bweights, 2, pts_index).unsqueeze(dim = -1)
        #write in n x b x smax 
        local_coords = self.calculate_local_coords(pts_hit, batch_nodes_T, nodes_weights, j_transformed, body, poses, shapes, R, Th) 
        #n x b x s_max x 256
        f_hit = self.Feature_field(ei, xyz_embedder(local_coords), s_max)
        f_ = f_hit * bweights_
        f_blend = torch.zeros(batch_size, pts_num, 256, device = f_hit.device)
        self.cuda_module.launch_blend_feature(f_, pts_index, f_blend, 256, nodes_n, batch_size, pts_num, s_max)
        torch.cuda.synchronize(pts_index.device)
        bweights += 0.00000001
        bweights = bweights.permute(1, 2, 0)
        weights_sum = torch.sum(bweights, dim = -1, keepdim = True)
        f_blend = f_blend / weights_sum
        c, d = self.Nerf(f_blend, view_embedder(wviewdir))
        return c, d, ei, nodes_delta, mean, logvar
