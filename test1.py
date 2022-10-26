import numpy as np
from torch.utils.data.dataloader import DataLoader
import cv2
import os
import torch

from lib.utils.write_ply import write_ply

from lib.datasets.light_stage.multi_view_dataset_msk import Dataset


kintree =  [[1, 0], [2, 1], [3, 2], [4, 3], [5, 1], [6, 5], [7, 6], [8, 1],
                [9, 8], [10, 9], [11, 10], [12, 8], [13, 12], [14, 13],
                [15, 0], [16, 0], [17, 15], [18, 16], [19, 14], [20, 19],
                [21, 14], [22, 11], [23, 22], [24, 11]]

distance = np.array(25, 25)
for i in range(0, 24):
    for j in range(0, 24):
        

data_root = 'data/zju_mocap/CoreView_387'
human = 'CoreView_387'
ann_file = 'data/zju_mocap/CoreView_387/annots.npy'
split = 'test'



dataset = Dataset(data_root, human, ann_file, split)
data_loader = DataLoader(dataset, batch_size=1)

result_dir = os.path.join('comparision')
pth1 = os.path.join('ply/ray_o.ply')
pth2 = os.path.join('ply/ray_d.ply')
for i, batch in enumerate(data_loader):
    image = torch.ones([512, 512, 3])
    rgb = batch['rgb'][0]
    ray_o = batch['ray_o'][0]
    ray_d = batch['ray_d'][0]
    near = batch['near']
    far = batch['far']
    frame_index = batch['frame_index'].item()
    m = batch['msk'][0].reshape(512,512)
    image[m] = rgb
    cv2.imwrite('a.png', image.detach().numpy()[..., [2, 1, 0]] * 255)
     

    
            # np.savetxt('mask{}.txt'.format(i), mask)

# ray_o = data['ray_o']
# ray_d = data['ray_d']
# sh = ray_o.shape #wpts: batchs x ray_o x N_sampled x 3
# max_range = 1
# poses = data['params']['poses']
# betas = data['params']['shapes']
# Rh = data['R']
# Th = data['Th']
# betas = np.squeeze(betas, axis = 0)
# # poses = np.squeeze(poses, axis = 0)
# # s_p = os.path.join('ply/test.ply')
# def to_cuda(batch, device):
#     for k in batch:
#         if k == 'meta':
#             continue
#         if isinstance(batch[k], tuple) or isinstance(batch[k], list):
#             batch[k] = [b.to() for b in batch[k]]
#         if isinstance(batch[k], dict):
#             for key in batch[k]:
#                 batch[k][key] = batch[k][key].to(device)
#         else:
#             batch[k] = batch[k].to(device)
#     return batch

# body = BodyModel(os.path.join('smpl_model'), 128, betas, gender ='male')
# network = Network()
# net_wrapper = NetworkWrapper(network)
# ret, loss, scalar_stat, image_stat = net_wrapper(to_cuda(data, device))
# print(scalar_stat)

# # nodes_T = body.basis['v_shaped'][body.basis['nodes_ind']]
# # batch_nodes = nodes_T.unsqueeze(dim = 0)
# # nodes_weights = body.basis['weights'][body.basis['nodes_ind']]
# # batch_nodes_posed, j_transformed = body.get_lbs(poses, betas, body.basis['weights'][body.basis['nodes_ind']], batch_nodes, Rh = Rh, Th = Th)
# # w = network.blend_weights(wpts.view(1, -1, 3), batch_nodes_posed)
# # torch.set_printoptions(profile='full')
# # print(w)
    

# # t = torch.ones(1, 1)
# # mean, std = network.encode(t, poses)
# # z = network.GaussianSample(mean, std)
# # ei, delta = network.decode(z, poses)
# # print(delta.shape)
# # bweights = network.blend_weights(wpts.view([1,-1,3]), batch_nodes_posed)
# #B x V x nodes x 3
# # local_coords = network.calculate_local_coords(wpts.view([1,-1,3]), batch_nodes_posed, nodes_weights, j_transformed, body, poses, betas, Rh, Th).squeeze()
# # for i in range(128):
#     # write_ply(os.path.join('ply/{}.ply'.format(i)), local_coords[:, i, :])
# # batch_nodes_T, j_T = body.get_lbs(poses, betas, body.basis['weights'][body.basis['nodes_ind']], batch_nodes_posed, Rh = Rh, Th = Th, joints = j_transformed, inverse = True)


# # fig = plt.figure(figsize = [5, 5])
# # ax1 = fig.add_subplot(221, projection = '3d')
# # ax2 = fig.add_subplot(222, projection = '3d')
# # pts = body.basis['J_shaped'].transpose(0, 1)

# # for i, j in enumerate(parents):
# #     ax.plot([pts[0][i], pts[0][j]], [pts[1][i], pts[1][j]],
# #             [pts[2][i], pts[2][j]],
# #             lw=1,
# #             color=kintree['color'][i],
# #             alpha=1)
# # ax1.set_xlim(-max_range, max_range)
# # ax1.set_ylim(-max_range, max_range)
# # ax1.set_zlim(-1, 1)
# # ax2.set_xlim(-max_range, max_range)
# # ax2.set_ylim(-max_range, max_range)
# # ax2.set_zlim(-1, 1)
# # x = body.basis['v_shaped'][:, 0]
# # y = body.basis['v_shaped'][:, 1]
# # z = body.basis['v_shaped'][:, 2]
# # x_n = batch_nodes_T[:,:, 0]
# # y_n = batch_nodes_T[:,:, 1]
# # z_n = batch_nodes_T[:,:, 2]

# # x2_j = j_transformed[:,:, 0]
# # y2_j = j_transformed[:,:, 1]
# # z2_j = j_transformed[:,:, 2]
# # x2_n = batch_nodes_posed[:, :, 0]
# # y2_n = batch_nodes_posed[:, :, 1]
# # z2_n = batch_nodes_posed[:, :, 2]

# # ax1.scatter(x_n, y_n,z_n, s = 15, c = 'green')
# # ax2.scatter(x2_n,y2_n,z2_n, s = 15, c = 'green')
# # fig.show()
# # input()


