#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_multiplication(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N) {
    const unsigned int g_i = get_global_id(0);
    const unsigned int g_j = get_global_id(1);
    const unsigned int l_i = get_local_id(0);
    const unsigned int l_j = get_local_id(1);
    __local float tile_a[TILE_SIZE * TILE_SIZE];
    __local float tile_b[TILE_SIZE * TILE_SIZE];

    float sum = 0;

    for (int tile_k = 0; tile_k * TILE_SIZE < K; ++tile_k) {
        int y = tile_k * TILE_SIZE + l_i;
        if (g_j < M && y < K) {
            tile_a[l_j * TILE_SIZE + l_i] = as[g_j * K + y];
        } else {
            tile_a[l_j * TILE_SIZE + l_i] = 0;
        }
        int x = tile_k * TILE_SIZE + l_j;
        if (g_i < N && x < K) {
            tile_b[l_j * TILE_SIZE + l_i] = bs[x * N + g_i];
        } else {
            tile_b[l_j * TILE_SIZE + l_i] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[l_j * TILE_SIZE + k] * tile_b[k * TILE_SIZE + l_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (g_j < M && g_i < N) {
        cs[g_j * N + g_i] = sum;
    }
}
