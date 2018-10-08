#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum_first(__global const int* xs,
                     unsigned int n,
                     __global int* sum,
                     __global int* max_sum,
                     __global int* max_pos) {
    uint g_id = get_global_id(0);

    uint l_id = get_local_id(0);
    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[l_id] = xs[g_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_id == 0) {
        int s = 0;
        int mx = 0;
        int pos_mx = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            s += local_xs[i];
            if (s > mx) {
                mx = s;
                pos_mx = i + 1;
            }
        }
        sum[g_id / WORK_GROUP_SIZE] = s;
        max_sum[g_id / WORK_GROUP_SIZE] = mx;
        max_pos[g_id / WORK_GROUP_SIZE] = g_id + pos_mx;
    }
}

__kernel void max_prefix_sum(__global const int* in_sum,
                             __global const int* in_max_sum,
                             __global const int* in_max_pos,
                             unsigned int n,
                             __global int* sum,
                             __global int* max_sum,
                             __global int* max_pos) {
    uint g_id = get_global_id(0);

    uint l_id = get_local_id(0);
    __local int local_sum[WORK_GROUP_SIZE];
    local_sum[l_id] = in_sum[g_id];

    __local int local_max_sum[WORK_GROUP_SIZE];
    local_max_sum[l_id] = in_max_sum[g_id];

    __local int local_max_pos[WORK_GROUP_SIZE];
    local_max_pos[l_id] = in_max_pos[g_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_id == 0) {
        int s = 0;
        int mx = 0;
        int pos_mx = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            if (s + local_max_sum[i] > mx) {
                mx = s + local_max_sum[i];
                pos_mx = local_max_pos[i];
            }
            s += local_sum[i];
        }
        sum[g_id / WORK_GROUP_SIZE] = s;
        max_sum[g_id / WORK_GROUP_SIZE] = mx;
        max_pos[g_id / WORK_GROUP_SIZE] = pos_mx;
    }
}
