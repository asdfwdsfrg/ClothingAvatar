#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <torch/extension.h>
using namespace torch::indexing;


__device__ void lock(int *mutex){
    while(atomicCAS(mutex, 0, 1) != 0);
}

__device__ void unlock(int *mutex){
    atomicExch(mutex, 0);
}


__global__ void get_pts_hit(float *pts, bool *mask, float *hits_pts, int iter, int n_nodes, int batches, int pts_num, int s_max)
{
    __shared__ int index;
    int nodes_ind = blockIdx.x;
    int batch_ind = blockIdx.y;
    int pts_ind = iter * 1024 + threadIdx.x;
    __shared__ int mutex;
    mutex = 0;
    if(threadIdx.x == 0)
    {
        index = 0;
    }
    __syncthreads();
    if(pts_ind < pts_num && mask[nodes_ind * batches * pts_num + batch_ind * pts_num + pts_ind] == true)
    {
        int ind = nodes_ind * batches * pts_num * 4 + batch_ind * pts_num * 4 + pts_ind * 4;
        int hit_ind = nodes_ind * batches * s_max * 4 + batch_ind * s_max * 4 ;
        lock(&mutex);
        hit_ind = hit_ind + index * 4;
        hits_pts[hit_ind] = pts[ind];
        hits_pts[hit_ind + 1] = pts[ind + 1];
        hits_pts[hit_ind + 2] = pts[ind + 2];
        hits_pts[hit_ind + 3] = pts[ind + 3];
        index = index + 1;
        unlock(&mutex);
    }
}


void pts_hit(float *pts, bool *mask, float *hits_pts, int n_nodes, int batches, int pts_num, int s_max)
{
//pts: n x B x v x 3
    
    dim3 block(1024);
    dim3 grid(n_nodes, batches);
    int iter = pts_num / 1024;
    // std::cout << pts[0] << std::endl;
    // float *gpu_pts, *mask, *hits_pts;
    // cudaMalloc((float **)&gpu_pts, n_nodes * batches * pts_num * 3 * sizeof(float));
    for(int i = 0; i < iter; i++)
    {
        get_pts_hit<<<grid, block>>>(pts, mask, hits_pts, i, n_nodes, batches, pts_num, s_max);
    }
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

__global__ void put_pts(float *f, int *ind, float *f_target, int f_dim, int iter, int n_nodes, int batches, int pts_num, int s_max)
{
    int64_t nodes_ind = blockIdx.x;
    int64_t batch_ind = blockIdx.y;
    int64_t i = blockIdx.z;
    
    int64_t pts_ind = iter * 1024 + threadIdx.x;
    if(pts_ind < s_max)
    {
        int64_t source_i = nodes_ind * batches * s_max * f_dim + batch_ind * s_max * f_dim + pts_ind * f_dim;
        int64_t ind_i = nodes_ind * batches * s_max + batch_ind * s_max + pts_ind;
        int64_t target_i = nodes_ind * batches * pts_num * f_dim + batch_ind * pts_num * f_dim + ind[ind_i] * f_dim;
        f_target[target_i + i] = f[source_i + i];
    }
}


void pts_put(float *f, int *ind, float *f_target, int f_dim, int n_nodes, int batches, int pts_num, int s_max)
{
//N x B x s_max x 256 -> N x B x V x 256
    dim3 block(1024);
    dim3 grid(n_nodes, batches, 256);
    int iter = s_max / 1024;

    for(int i = 0; i < iter; i++)
    {
        put_pts<<<grid, block>>>(f, ind, f_target, f_dim, i, n_nodes, batches, pts_num, s_max);
    }
}