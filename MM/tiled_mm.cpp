/* Solution of matrix multiplication in CUDA with tiling */

#include<iostream>
#include<random>
#include<stdlib.h>

#define TILE_DIM 16

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline uint64_t current_time_nsecs()
{
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

__global__ void tiled_mm(float *A, float *B, float *C, int N, int M, int R)
{
    __shared__ float A_smem[TILE_DIM][TILE_DIM];
    __shared__ float B_smem[TILE_DIM][TILE_DIM];

    int ix = (blockIdx.x * blockDim.x) + threadIdx.x;
    int iy = (blockIdx.y * blockDim.y) + threadIdx.y;

    float val = 0;

    for (int t=0; t<(std::ceil(((float) M)/TILE_DIM)); t++) {
        if (iy < N && (t * TILE_DIM) + threadIdx.x < M) {
            A_smem[threadIdx.y][threadIdx.x] = A[(iy * M) + (t * TILE_DIM) + threadIdx.x];
        }
        else {
            A_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (ix < R && (t * TILE_DIM) + threadIdx.y < M) {
            B_smem[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y)*R + ix];
        }
        else {
            B_smem[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // compute C tile
        for (int k=0; k<TILE_DIM; k++) {
            val += A_smem[threadIdx.y][k] * B_smem[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (ix < R && iy < N) {
        // update C in global memory
        C[iy*R + ix] = val;
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <N> <M> <R>" << std::endl;
        exit(1);
    }

    int N = atoi(argv[1]);
    int M = atoi(argv[2]);
    int R = atoi(argv[3]);

    // allocation of the host arrays
    float *host_A = (float *) malloc(sizeof(float) * N * M);
    float *host_B = (float *) malloc(sizeof(float) * M * R);
    float *host_C = (float *) malloc(sizeof(float) * N * R);

    // initialization of the host arrays
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    for (int i=0; i<N*M; i++) {
        host_A[i] = dis(gen);
    }

    for (int i=0; i<M*R; i++) {
        host_B[i] = dis(gen);
    }

    // allocate GPU arrays
    cudaSetDevice(0); // set the working device
    float *dev_A;
    float *dev_B;
    float *dev_C;
    gpuErrchk(cudaMalloc((void**) &dev_A, N*M*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &dev_B, M*R*sizeof(float)));
    gpuErrchk(cudaMalloc((void**) &dev_C, N*R*sizeof(float)));

    uint64_t initial_time = current_time_nsecs();

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, N*M*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_B, host_B, M*R*sizeof(float), cudaMemcpyHostToDevice));

    uint64_t initial_time2 = current_time_nsecs();

    // perform computation on GPU
    dim3 gridDim(std::ceil(((float) R)/TILE_DIM), std::ceil(((float) N)/TILE_DIM));
    dim3 blockDim(TILE_DIM, TILE_DIM);
    tiled_mm<<<gridDim, blockDim>>>(dev_A, dev_B, dev_C, N, M, R);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    uint64_t end_time2 = current_time_nsecs();
    uint64_t elapsed2 = end_time2 - initial_time2;
    std::cout << "Kernel time: " << ((float) elapsed2)/1000.0 << " usec" << std::endl;

    // copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_C, dev_C, N*R*sizeof(float), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed)/1000.0 << " usec" << std::endl;

    // deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));
    gpuErrchk(cudaFree(dev_C));

    // check results
    float *check_C = (float *) malloc(sizeof(float) * N * R);
    for (int i=0; i<N; i++) {
        for (int j=0; j<R; j++) {
            float v = 0;
            for (int k=0; k<M; k++) {
                v += host_A[i * M + k] * host_B[k * R + j];
            }
            check_C[i * R + j] = v;
        }
    }
    for (int i=0; i<N*R; i++) {
        float diff = std::abs(check_C[i] - host_C[i]);
        if (diff > 0.5e-1) {
            std::cout << "Result error!" << std::endl;
            abort();
        }
    } 
    std::cout << "Result is ok!" << std::endl;

    // deallocate host memory
    free(host_A);
    free(host_B);
    free(host_C);
    free(check_C);

    return 0;
}
