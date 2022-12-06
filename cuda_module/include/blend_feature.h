void pts_hit(float *pts, bool *mask, float *hits_pts, int n_nodes, int batches, int pts_num, int s_max);
void pts_put(float *f, int64_t *ind, float *f_target, int f_dim, int n_nodes, int batches, int pts_num, int s_max);
void blend_feature(float *f_hit, int64_t *index, float *f_blend, int f_dim, int n_nodes, int batches, int pts_num, int s_max);

