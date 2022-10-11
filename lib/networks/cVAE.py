import os
from random import randrange
from threading import local
import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.networks.body_model import BodyModel 
from lib.networks.embedder import *
from lib.config import cfg
from lib.utils.write_ply import write_ply
from . import embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        #encoder
        
        self.ec_ly1 = nn.ModuleList([nn.Linear(72 + time_dim, 64) for i in range(cfg.n_nodes)])
        self.ec_ly21= nn.ModuleList([nn.Linear(64, 8) for i in range(cfg.n_nodes)])
        self.ec_ly22= nn.ModuleList([nn.Linear(64, 8) for i in range(cfg.n_nodes)])
       
        #decoder
        self.dc_ly1= nn.ModuleList([nn.Linear(80, 64) for i in range(cfg.n_nodes)])
        self.dc_ly21= nn.ModuleList([nn.Linear(64, 32) for i in range(cfg.n_nodes)])
        self.dc_ly22= nn.ModuleList([nn.Linear(64, 3) for i in range(cfg.n_nodes)])
        self.actvn = nn.ReLU()
        self.actvn2 = nn.Sigmoid()

        #feature field
        self.f_ly1= nn.ModuleList([nn.Linear(32 + xyz_dim, 64) for i in range(cfg.n_nodes)])
        self.f_ly2= nn.ModuleList([nn.Linear(64, 64) for i in range(cfg.n_nodes)])
        self.f_ly3= nn.ModuleList([nn.Linear(64, 64) for i in range(cfg.n_nodes)])
        self.f_ly4= nn.ModuleList([nn.Linear(64, 256) for i in range(cfg.n_nodes)])

        #Nerf color
        self.c_ly1 = nn.Linear(256, 64)
        self.c_ly2 = nn.Linear(64, 3)

        #NerF density
        self.d_ly1 = nn.Linear(256, 1)

        
        
    def encode(self, t_ped, pose):
        '''
        poses: batch x 72
        t_ped: batch x time_dim
        mean, std:  Batchs x nodes x 8
        '''
        #batchs x 73
        encoder_in = torch.cat((torch.tensor(t_ped.expand(pose.shape[0], time_dim), dtype=pose.dtype), pose.view(-1, 72)), dim = -1)
        #nodes x batchs x 64
        net = [self.actvn(self.ec_ly1[node_i](encoder_in)) for node_i in range(cfg.n_nodes)]
        #nodes x batchs x 8 
        mean = torch.stack([self.ec_ly21[node_i](net[node_i]) for node_i in range(cfg.n_nodes)], dim = 0)
        std = torch.stack([(self.actvn2(self.ec_ly22[node_i](net[node_i]))) for node_i in range(cfg.n_nodes)], dim = 0)
        
        return mean, std
    
    def decode(self, z, poses):
        '''
        poses: Batchs x 1 x 72
        z: nodes x Batchs x 8
        ei: Batchs x nodes x 32
        '''
        #nodes x Batchs x 72
        poses = torch.squeeze(poses).expand(cfg.n_nodes, poses.shape[0], poses.shape[-1])
        #nodes x Batchs x 80
        input = torch.cat([z, poses], dim = -1)
        #nodes x batchs x 64
        net = [self.actvn(self.dc_ly1[node_i](input[node_i])) for node_i in range(cfg.n_nodes)]
        ei = torch.stack([self.dc_ly21[node_i](net[node_i]) for node_i in range(cfg.n_nodes)],dim = 0)
        delta_ni = torch.stack([self.dc_ly22[node_i](net[node_i]) for node_i in range(cfg.n_nodes)], dim = 0)
        #nodes x batches x (32, 3)
        return ei, delta_ni
    
    def Feature_field(self, ei, local_coords):
        '''
        local_coords: B x V x nodes X 3
        ei:nodes x B x 32
        '''
        ei = torch.transpose(torch.transpose(ei.expand(local_coords.shape[1], local_coords.shape[-2], local_coords.shape[0], ei.shape[-1]), 0, 1), 0, 2)
        #B x V x N x 35
        pts_in = torch.cat((ei, local_coords), dim = -1)
        net = [self.actvn(self.f_ly1[node_i](pts_in[:, :, node_i, :])) for node_i in range(cfg.n_nodes)]
        net = [self.actvn(self.f_ly2[node_i](net[node_i])) for node_i in range(cfg.n_nodes)]
        net = [self.actvn(self.f_ly3[node_i](net[node_i])) for node_i in range(cfg.n_nodes)]
        f = torch.stack([self.f_ly4[node_i](net[node_i]) for node_i in range(cfg.n_nodes)], dim = 0)
        #nodes x B x V x 256
        return f
        
    def GaussianSample(self, mean, std):
        #z: nodes x batchs x 8
        return torch.normal(mean, std)

    def blend_feature(self, bweights, f):
        '''
        bweights: B x V x nodes 
        f: nodes x v x B x256
        '''
        #B x V x  256 x nodes
        f = (torch.transpose(torch.transpose(f, 0, 2), 2, 3))
        #B x V x nodes x 1
        bweights = bweights.unsqueeze(dim = -1)
        bweights += torch.tensor(0.000001)
        f_blend = torch.matmul(f, bweights).squeeze(dim = -1)
        weights = torch.sum(bweights, dim = -2)
        #B X V X 1
        #B x V x 256

        return f_blend / weights

    
    def Nerf(self, f):
        '''
        f: B x V x 256
        '''
        netc = self.actvn(self.c_ly1(f))
        c = self.c_ly2(netc)
        d = self.actvn(self.d_ly1(f))
        return c, d

    def pts_to_can_pts(self, pts, sp_input):
        """transform pts from the world coordinate to the smpl coordinate"""
        Th = sp_input['Th']
        pts = pts - Th
        R = sp_input['R']
        pts = torch.matmul(pts, R)
        return pts

    def nodes_transform(self, nodes_T, nodes_weights, params, Rh, Th):
        """transform node coord from T to pose"""
        nodes_posed = self.smpl(params['poses'], params['shapes'], Rh = Rh, Th = Th)
        return nodes_posed 


    def pts_to_node_pts(self, sampled_pts, params, Rh, Th, nodes_posed, nodes_T, nodes_delta):
        """get the node index and coordinate inside node"""
        
        pts_pose_to_T = self.smpl(params['poses'], params['shapes'], Rh = Rh, Th = Th, inverse = True) 
        node_pts = pts_pose_to_T - nodes_T - nodes_delta 
        return node_pts 
        

    def blend_weights(self, wpts, nodes_posed):
        """feature blending weights"""    
        '''
            wpts:B x V x 3
            nodes_posed: B x nodes_n x 3
            return:
                bweights: B x V x nodes_n  the nodes influence on each vertex
        '''
        bweights = torch.zeros([wpts.shape[0], wpts.shape[1], nodes_posed.shape[1]], device=wpts.device)
        for i in range(nodes_posed.shape[1]):
            nodes_influ = torch.exp(-(torch.sum(torch.pow((wpts - nodes_posed[:,i,:]), 2), dim = 2) / torch.pow(torch.tensor(cfg.sigma), 2))) - torch.tensor(cfg.epsilon)
            bweights[:,:,i] = torch.max(nodes_influ, torch.tensor(0).float())
        return bweights


    def calculate_local_coords(self, wpts, nodes_T, nodes_delta, nodes_weights, J, body, poses, shapes, R, Th):
        """get local_coords"""
        '''
            wpts: B x V x 3
            nodes: B x nodes_n x 3
            nodes_delta: nodes_n x B x 3
            nodes_weights: B x nodes_n x Joints
            return:
                local_coords: B x V x nodes_n x 3
                local_coords[:,v, i, :] means point v's coord in i nodes 
        '''
        #B x V x nodes_n x 3
        wpts = torch.transpose(torch.transpose(wpts.expand(nodes_T.shape[1],wpts.shape[0], wpts.shape[1], 3), 0, 1), 1, 2)
        #B X V X nodes_n x 3
        coords_T, j_transformed = body.get_lbs(poses, shapes, nodes_weights, wpts, joints = J, Rh = R, Th = Th, inverse = True)
        coords_T = coords_T.unsqueeze(dim = 1)
        #V x B x nodes_n x 3
        local_coords = torch.transpose(coords_T, 0, 1) - nodes_T - torch.transpose(nodes_delta, 0, 1)
        return torch.transpose(local_coords, 0, 1) 
        

    def forward(self, input, wpts, body):
        #transform sampled points from world coord to smpl coord
        wpts = self.pts_to_can_pts(wpts, input)
        params = input['params']
        poses = params['poses']
        shapes = params['shapes']
        batch_size = input['batch_size']
        R = input['R']
        Th = input['Th']
        nodes_ind = body.basis['nodes_ind']
        #nodes in T pose
        nodes_T = body.basis['v_shaped'][nodes_ind]
        batch_nodes_T = nodes_T.unsqueeze(dim = 0)
        nodes_weights = body.basis['weights'][nodes_ind]
        batch_nodes_posed, j_transformed = body.get_lbs(poses, shapes, nodes_weights, batch_nodes_T, Rh = R, Th = Th)
        wpts = wpts.view(1, -1, 3)
        bweights = self.blend_weights(wpts, batch_nodes_posed) #TODO high complexity
        t_ped = time_embedder(input['latent_index'])
        mean, std = self.encode(t_ped, params['poses'])
        if self.training:
            z = self.GaussianSample(mean, std)
        elif cfg.mod == 'novel':
            z = torch.zeros_like(mean)
        else:
            z = mean
        ei, nodes_delta = self.decode(z, params['poses'])  
        #B x V x Nodes x 3
        local_coords = self.calculate_local_coords(wpts, batch_nodes_T, nodes_delta, nodes_weights, j_transformed, body, poses, shapes, R, Th) #TODO High complexity
        #nodes x B x V x 256k
        f = self.Feature_field(ei, xyz_embedder(local_coords)).to(bweights.device)
        f_blend = self.blend_feature(bweights, f)
        #batchs x v x 3,1
        c, d = self.Nerf(f_blend)
        #ei, nodes_delta: nodes x batches x (32, 3)
        #mean, std: nodes x batchs x 8 
        return c, d, ei, nodes_delta, mean, std

