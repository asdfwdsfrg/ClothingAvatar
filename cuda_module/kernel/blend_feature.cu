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
    __shared__ int mutex;
    mutex = 0;
    for(int i = 0;  i < iter; i++)
    {
        int pts_ind = i * 1024 + threadIdx.x;
        if(pts_ind == 0)
        {
            index = 0;
        }
        __syncthreads();
        if(pts_ind < pts_num && mask[nodes_ind * batches * pts_num + batch_ind * pts_num + pts_ind] == true)
        {
            int ind = nodes_ind * batches * pts_num * 4 + batch_ind * pts_num * 4 + pts_ind * 4;
            int hit_ind = nodes_ind * batches * s_max * 4 + batch_ind * s_max * 4;
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
}

void pts_hit(float *pts, bool *mask, float *hits_pts, int n_nodes, int batches, int pts_num, int s_max)
{
//pts: n x B x v x 3
    dim3 block(1024);
    dim3 grid(n_nodes, batches);
    int iter = pts_num / 1024 + 1;
    get_pts_hit<<<grid, block>>>(pts, mask, hits_pts, iter, n_nodes, batches, pts_num, s_max);
}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

__global__ void blend_feature_kernel(float *f_hit, int64_t *index, float *f_blend, int f_dim, int n_nodes, int batches, int pts_num, int s_max, int iter)
{
    int64_t node_ind = blockIdx.x;
    int64_t batch_ind = blockIdx.y;
    int64_t pts_ind = 1024 * iter + threadIdx.x;
    for(int k = 0; k < iter; k++)
    {
        int64_t pts_ind = 1024 * k + threadIdx.x;
        if(pts_ind < s_max)
        {
            int64_t source_i = node_ind * batches * s_max * f_dim + batch_ind * s_max * f_dim + pts_ind * f_dim;
            int64_t ind = node_ind * batches * s_max + batch_ind * s_max + pts_ind;
            int64_t target_i = batch_ind * pts_num * f_dim + index[ind] * f_dim;
            if(index[ind] == -1)
            {
                return;
            }
            for(int i = 0; i < f_dim; i++)
            {
                f_blend[target_i] = f_blend[target_i] + f_hit[source_i];
            }
        }
    }
}

void blend_feature(float *f_hit, int64_t *index, float *f_blend, int f_dim, int n_nodes, int batches, int pts_num, int s_max)
{
    //f_hit: n x b x s_max x 256
    //index: n x b x s_max 
    //return: f_blend b x v x 256
    dim3 block(1024);
    dim3 grid(n_nodes, batches);
    int iter = s_max / 1024 + 1;
    blend_feature_kernel<<<grid, block>>>(f_hit, index, f_blend, f_dim, n_nodes, batches, pts_num, s_max, iter);
}

__global__ void put_pts(float *f, int64_t *ind, float *f_target, int f_dim, int iter, int n_nodes, int batches, int pts_num, int s_max)
{
    int64_t nodes_ind = blockIdx.x;
    int64_t batch_ind = blockIdx.y;
    // int64_t i = blockIdx.z;
    for(int k = 0; k < iter; k++)
    {
        int64_t pts_ind = k * 1024 + threadIdx.x;
        if(pts_ind < s_max)
        {
            int64_t source_i = nodes_ind * batches * s_max * f_dim + batch_ind * s_max * f_dim + pts_ind * f_dim;
            int64_t ind_i = nodes_ind * batches * s_max + batch_ind * s_max + pts_ind;
            int64_t target_i = nodes_ind * batches * pts_num * f_dim + batch_ind * pts_num * f_dim + ind[ind_i] * f_dim;
            for(int i = 0; i < f_dim; i++)
            {
                f_target[target_i + i] = f[source_i + i];
            }
        }
    }
}

void pts_put(float *f, int64_t *ind, float *f_target, int f_dim, int n_nodes, int batches, int pts_num, int s_max)
{
    //N x B x s_max x 256 -> N x B x V x 256
    dim3 block(1024);
    dim3 grid(n_nodes, batches);
    int iter = s_max / 1024 + 1;
    put_pts<<<grid, block>>>(f, ind, f_target, f_dim, iter, n_nodes, batches, pts_num, s_max);
}