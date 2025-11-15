/* Example of using tensor core with MMA instructions for D = A*B+C and 16x8x8 */

#include<vector>
#include<iostream>
#include<random>
#include<stdlib.h>

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void tensor_16x8_kernel(float *A, float *B, float *C, float *D)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < 32) {
        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
            : "=f"(D[id<<1]),
              "=f"(D[(id<<1) + 1]),
              "=f"(D[(id<<1) + 64]),
              "=f"(D[(id<<1) + 65])
            : "r"(*reinterpret_cast<uint32_t const *>(&A[8*(id>>2) + (id&3)])),
              "r"(*reinterpret_cast<uint32_t const *>(&A[64 + 8*(id>>2) + (id&3)])),
              "r"(*reinterpret_cast<uint32_t const *>(&A[4 + 8*(id>>2) + (id&3)])),
              "r"(*reinterpret_cast<uint32_t const *>(&A[68 + 8*(id>>2) + (id&3)])),
              "r"(*reinterpret_cast<uint32_t const *>(&B[(id>>2) + (id&3)*8])),
              "r"(*reinterpret_cast<uint32_t const *>(&B[32+(id>>2) + (id&3)*8])),
              "f"(C[id<<1]),
              "f"(C[(id<<1) + 1]),
              "f"(C[(id<<1) + 64]),
              "f"(C[(id<<1) + 65])
        );
    }
}

int main(int argc, char **argv)
{
    // allocation and initialization of the host vectors
    auto host_A = std::vector<float>{ // A is 16x8
        -0.029,  2.105, -1.886,  0.295, -0.114,  0.181,  0.889,  0.996,
        -0.045, -0.544,  0.583, -0.853, -1.933, -1.307, -0.455, -0.192,
        -0.971, -1.172, -0.597, -2.666, -0.322,  1.346, -0.027,  0.411,
         0.436, -1.798,  0.362, -0.899,  0.176, -0.616, -0.230,  0.486,
        -0.304, -0.406, -0.001, -1.545,  1.060,  1.139,  1.473,  0.522,
        -1.135, -0.768,  1.767,  1.431, -0.728,  0.155, -1.767, -1.696,
         1.407, -1.319,  0.601,  1.388, -1.371, -0.532,  0.004,  1.250,
         0.850, -0.154, -1.392,  0.170, -1.029,  1.483,  0.023,  0.225,
        -0.185,  1.259,  1.232,  0.600,  0.099,  0.080,  1.711, -1.145,
         0.010, -1.740, -1.447, -0.055,  0.306,  0.794,  0.611, -1.414,
         1.035, -0.164,  0.523, -1.527, -0.028,  1.389,  0.404,  1.185,
        -0.434, -0.201, -0.580, -1.135,  1.136, -1.856, -0.687, -0.285,
         0.145, -1.240,  0.017,  0.566, -0.148, -0.640, -1.786,  0.334,
         0.071, -0.002, -1.080, -0.769,  1.097, -0.233, -0.210,  1.345,
        -1.869,  0.477,  1.172,  0.433,  2.041, -0.187,  0.025, -0.158,
        -0.587, -0.525, -1.792, -0.261,  1.541, -0.011,  0.527, -0.814
    };
    auto host_B = std::vector<float>{ // B is 8x8
         0.151, -0.551,  0.251,  0.594, -0.380,  0.119,  0.035, -0.814,
        -0.008, -0.017,  0.356,  0.609, -0.129, -0.311, -0.193, -0.213,
        -0.678,  0.153,  0.628,  0.265,  0.272,  0.657, -0.238, -0.130,
         0.582, -0.177, -0.392,  0.219,  0.246, -0.036,  0.132,  0.200,
        -0.084,  0.004,  0.251,  0.742, -0.296, -0.047,  0.000,  0.235,
         0.087,  0.094,  0.147, -0.448, -0.046,  0.322, -0.108, -0.011,
        -0.057, -0.006, -0.546, -0.200,  0.568,  0.115,  0.095, -0.390,
         0.053, -0.003,  0.045, -0.101,  0.494, -0.305,  0.722,  0.349
    };
    auto host_C = std::vector<float>{ // C is 16x8
        -0.077, -0.377,  0.367,  0.196,  0.196, -0.492,  0.076,  0.061,
        -0.080, -0.575,  0.582, -0.312,  1.398, -0.974,  0.702,  0.149,
         0.286,  1.697,  1.222,  1.819,  0.535,  0.485,  0.400,  1.442,
        -0.965,  1.804, -2.018, -0.115,  0.445, -0.073, -0.926,  2.370,
        -0.492,  1.280, -0.754,  0.718, -0.559, -0.818, -2.285, -0.152,
        -1.667,  0.659, -0.357,  0.883,  0.643, -0.061,  1.028, -0.280,
         1.952, -1.909, -1.774,  0.315, -0.300,  0.558, -0.425,  0.237,
        -0.624, -2.727, -0.988, -1.164,  1.633,  1.442,  1.539,  0.159,
        -0.755, -1.314, -0.414,  1.452, -0.664, -0.282, -0.051,  1.162,
        -0.232, -1.735, -1.378, -1.172, -1.932,  0.403, -0.002,  0.282,
         0.904,  1.210,  0.532, -0.018, -0.609,  0.441,  1.047, -0.649,
        -0.896,  0.251, -0.363, -1.750,  1.364,  0.332,  0.552,  0.080,
        -0.788, -0.291,  1.004, -0.037, -0.064,  1.263, -0.552,  0.701,
         0.542, -1.023,  0.520, -1.334,  2.252,  0.453,  0.675, -0.316,
         0.404, -1.913,  0.261,  1.448,  0.687,  2.093, -0.149,  0.346,
        -0.537, -0.520, -0.283,  1.143,  0.350, -0.357, -1.604,  0.260
    };
    auto host_D = std::vector<float>(16*8, 0.f); // D is 16x8

    // allocate GPU arrays
    cudaSetDevice(0); // set the working device
    float *dev_A;
    float *dev_B;
    float *dev_C;
    float *dev_D;
    gpuErrchk(cudaMalloc((void**) &dev_A, sizeof(float) * 16 * 8));
    gpuErrchk(cudaMalloc((void**) &dev_B, sizeof(float) * 8 * 8));
    gpuErrchk(cudaMalloc((void**) &dev_C, sizeof(float) * 16 * 8));
    gpuErrchk(cudaMalloc((void**) &dev_D, sizeof(float) * 16 * 8));

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A.data(), sizeof(float) * 16 * 8, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_B, host_B.data(), sizeof(float) * 8 * 8, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_C, host_C.data(), sizeof(float) * 16 * 8, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_D, host_D.data(), sizeof(float) * 16 * 8, cudaMemcpyHostToDevice));

    // perform computation on GPU
    tensor_16x8_kernel<<<1, 32>>>(dev_A, dev_B, dev_C, dev_D);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_D.data(), dev_D, sizeof(float) * 16 * 8, cudaMemcpyDeviceToHost));

    // deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));
    gpuErrchk(cudaFree(dev_C));
    gpuErrchk(cudaFree(dev_D));

    // check the results
    float *check_D = (float *) malloc(sizeof(float) * 16 * 8);
    for (int i=0; i<16; i++) {
        for (int j=0; j<8; j++) {
            float tmp = 0;
            for (int k=0; k<8; k++) {
                tmp += host_A[i * 8 + k] * host_B[k * 8 + j];
            }
            check_D[i * 8 + j] = tmp + host_C[i * 8 + j];
        }
    }
    for (int i=0; i<16*8; i++) {
        float diff = std::abs(check_D[i] - host_D[i]);
        if (diff > 0.5e-1) {
            std::cout << "Result error!" << std::endl;
            std::cout << "check_D[i]: " << check_D[i] << " != host_D[i]: " << host_D[i] << std::endl;
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // ceallocate host memory
    free(check_D);

    return 0;
}
