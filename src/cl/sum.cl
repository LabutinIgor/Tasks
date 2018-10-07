#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum(__global const int* xs,
                     unsigned int n,
                     __global int* res)
{
    int localId = get_local_id(0);
    int globalId = get_global_id(0);

    __local int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = xs[globalId];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        int sum = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            sum += local_xs[i];
        }
        atomic_add(res, sum);
    }
}
