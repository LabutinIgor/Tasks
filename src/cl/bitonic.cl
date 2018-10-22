#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256


__kernel void bitonic(__global float* as,
                      unsigned int n,
                      unsigned int i,
                      unsigned int j)
{
    const unsigned int g_id = get_global_id(0);

    int d = 1 << (i - j);
    bool up = ((g_id >> i) & 2) == 0;
    if ((g_id & d) == 0 && ((g_id | d) < n) && (as[g_id] > as[g_id | d]) == up) {
        float t = as[g_id];
        as[g_id] = as[g_id | d];
        as[g_id | d] = t;
    }
}


__kernel void bitonic_local(__global float* as,
                      unsigned int n,
                      unsigned int i,
                      unsigned int j0)
{
    const unsigned int g_id = get_global_id(0);
    const unsigned int l_id = get_local_id(0);
    __local float local_as[WORK_GROUP_SIZE];

    if (g_id < n) {
        local_as[l_id] = as[g_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int j = j0; j <= i; j++) {
        int d = 1 << (i - j);
        bool up = ((g_id >> i) & 2) == 0;
        if ((g_id & d) == 0 && ((g_id | d) < n) && (local_as[l_id] > local_as[l_id | d]) == up) {
            float t = local_as[l_id];
            local_as[l_id] = local_as[l_id | d];
            local_as[l_id | d] = t;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_id < n) {
        as[g_id] = local_as[l_id];
    }
}
