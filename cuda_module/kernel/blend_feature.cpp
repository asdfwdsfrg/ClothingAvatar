#include <torch/extension.h>
#include "/home/wzx2021/ClothingAvatar/cuda_module/include/blend_feature.h"


void launch_pts_hit(torch::Tensor &pts, torch::Tensor &mask, torch::Tensor &hits_pts, int nodes_n, int batches, int pts_num, int s_max) 
{
    pts_hit(pts.data_ptr<float>(), mask.data_ptr<bool>(), hits_pts.data_ptr<float>(), nodes_n, batches, pts_num, s_max);
}

void launch_put_pts(torch::Tensor &f, torch::Tensor &ind, torch::Tensor &f_target, int f_dim, int nodes_n, int batches, int pts_num, int s_max)
{
    pts_put(f.data_ptr<float>(), ind.data_ptr<int>(), f_target.data_ptr<float>(), f_dim, nodes_n, batches, pts_num, s_max);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_pts_hit",
          &launch_pts_hit,
          "pts_hit kernel warpper");
    m.def("launch_put_pts", 
            &launch_put_pts,
            "pts_put kernel warpper");
}