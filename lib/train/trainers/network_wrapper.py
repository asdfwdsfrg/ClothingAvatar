import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks import nerf_renderer
from torch.nn import KLDivLoss 
from lib.networks.body_model import BodyModel
from lib.train import make_optimizer
import numpy as np
from torch.distributions import Normal, kl_divergence
import os



class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()
        params_path = os.path.join('data/zju_mocap/CoreView_387/new_params/0.npy')
        #init structure model
        betas = np.load(params_path, allow_pickle=True).item()['shapes']
        betas = torch.from_numpy(betas)
        device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.body = BodyModel(os.path.join('smpl_model'), 128, betas, gender = 'male', device = device)
        self.net = net  #xyz_latent
        self.renderer = nerf_renderer.Renderer(self.net, self.body)        
        self.img2mse = lambda x, y : torch.sum(torch.sum((x - y) ** 2, dim = 1),dim=0)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        image_stats = {}
        loss = 0
        mask = batch['mask_at_box']
        lambda_img_loss = cfg.lambda_img_loss
        lambda_trans = cfg.lambda_trans 
        lambda_ebd = cfg.lambda_ebd
        lambda_kl = cfg.lambda_kl
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        normal_pred = Normal(ret['mean'], ret['std'])
        normal_t = Normal(0, 1)

        kl_loss = kl_divergence(normal_pred, normal_t)
        kl_loss = torch.sum(kl_loss.view(-1, 1), dim = 0) 
        scalar_stats.update({'kl_loss': kl_loss})
        trans_loss = torch.norm(ret['delta_nodes']) ** 2
        scalar_stats.update({'trans_loss': trans_loss})
        ebd_loss = torch.norm(ret['embedding']) ** 2
        scalar_stats.update({'ebd_loss': ebd_loss})
        
        
        loss += lambda_img_loss * img_loss + lambda_ebd * ebd_loss + lambda_kl * kl_loss + lambda_trans * trans_loss

        if 'rgb0' in ret:
            img_loss0 = self.img2mse(ret['rgb0'], batch['rgb'])
            scalar_stats.update({'img_loss0': img_loss0})
            loss += img_loss0

        mask_at_box = batch['mask_at_box']
        image = ret['rgb_map'][mask_at_box]
        scalar_stats.update({'loss': loss})
        image_stats.update({'image': image}) 
        return ret, loss, scalar_stats, image_stats
