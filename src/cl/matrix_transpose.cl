#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* as,
                               __global float* bs,
                               unsigned int M,
                               unsigned int K) {
    __local float local_as[TILE_SIZE * TILE_SIZE];
    const unsigned int g_i = get_global_id(0);
    const unsigned int g_j = get_global_id(1);
    const unsigned int l_i = get_local_id(0);
    const unsigned int l_j = get_local_id(1);

    if (g_i < K && g_j < M) {
        local_as[l_j * TILE_SIZE + l_i] = as[g_j * K + g_i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (l_i < l_j) {
        float tmp = local_as[l_j * TILE_SIZE + l_i];
        local_as[l_j * TILE_SIZE + l_i] = local_as[l_i * TILE_SIZE + l_j];
        local_as[l_i * TILE_SIZE + l_j] = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int x = g_i - l_i + l_j;
    int y = g_j - l_j + l_i;
    if (x < K && y < M) {
       //printf("%d %d  %d %d\n", g_i, g_j, x, y);
       bs[x * M + y] = local_as[l_j * TILE_SIZE + l_i];
    }

}
