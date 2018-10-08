#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/max_prefix_sum_cl.h"

template<typename T>
void raiseFail(const T& a, const T& b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void calcMaxSumOnGpu(ocl::Kernel& sum, ocl::Kernel& sumFirst, int n, const gpu::gpu_mem_32i& as_gpu, int& s, int& res) {
    unsigned int workGroupSize = 128;

    gpu::gpu_mem_32i s_gpu;
    gpu::gpu_mem_32i max_sum_gpu;
    gpu::gpu_mem_32i max_pos_gpu;
    unsigned int res_size = ((n - 1) / workGroupSize + 1);
    s_gpu.resize(res_size * sizeof(int));
    max_sum_gpu.resize(res_size * sizeof(int));
    max_pos_gpu.resize(res_size * sizeof(int));

    unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
    sumFirst.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n,
                  s_gpu, max_sum_gpu, max_pos_gpu);

    std::vector<int> sum_all(res_size);
    std::vector<int> mx(res_size);
    std::vector<int> pos_mx(res_size);

    s_gpu.read(sum_all.data(), res_size * sizeof(int));
    max_sum_gpu.read(mx.data(), res_size * sizeof(int));
    max_pos_gpu.read(pos_mx.data(), res_size * sizeof(int));

    gpu::gpu_mem_32i in_s_gpu;
    gpu::gpu_mem_32i in_max_sum_gpu;
    gpu::gpu_mem_32i in_max_pos_gpu;
    in_s_gpu.resize(res_size * sizeof(int));
    in_max_sum_gpu.resize(res_size * sizeof(int));
    in_max_pos_gpu.resize(res_size * sizeof(int));
    while (res_size > 1) {
        n = res_size;
        res_size = ((n - 1) / workGroupSize + 1);
        in_s_gpu.write(sum_all.data(), n * sizeof(int));
        in_max_sum_gpu.write(mx.data(), n * sizeof(int));
        in_max_pos_gpu.write(pos_mx.data(), n * sizeof(int));

        global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        sum.exec(gpu::WorkSize(workGroupSize, global_work_size), in_s_gpu, in_max_sum_gpu, in_max_pos_gpu, n,
                 s_gpu, max_sum_gpu, max_pos_gpu);

        s_gpu.read(sum_all.data(), res_size * sizeof(int));
        max_sum_gpu.read(mx.data(), res_size * sizeof(int));
        max_pos_gpu.read(pos_mx.data(), res_size * sizeof(int));
    }

    s = mx[0];
    res = pos_mx[0];
}

int main(int argc, char** argv) {
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);
            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();

            ocl::Kernel sum(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum");
            sum.compile();

            ocl::Kernel add(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "sum_first");
            add.compile();

            gpu::gpu_mem_32i as_gpu;
            as_gpu.resize(n * sizeof(as[0]));
            as_gpu.write(as.data(), n * sizeof(as[0]));

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int s, res;
                calcMaxSumOnGpu(sum, add, n, as_gpu, s, res);
                EXPECT_THE_SAME(reference_max_sum, s, "GPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, res, "GPU result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
