/* Example of using PTX intrinsics in a CUDA kernel */

#include<iostream>
#include<stdlib.h>

#define N 1024
#define BLOCK_DIM 128

inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        std::cerr << "GPUassert code: " << cudaGetErrorString(code) << ", file: " << file << ", line: " << line << std::endl;
        exit(code);
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

__global__ void kernel(int *A, int *B, int *C)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (id < N) {
        asm("lop3.b32 %0, %1, %2, %3, 0b11101010;"
            : "=r"(C[id])
            : "r"(A[id]), "r"(B[id]), "r"(C[id])
        );
    }
}

int main(int argc, char **argv)
{
    // allocation of the host arrays
    int *host_A = (int *) malloc(sizeof(int) * N);
    int *host_B = (int *) malloc(sizeof(int) * N);
    int *host_C = (int *) malloc(sizeof(int) * N);
    int *host_check = (int *) malloc(sizeof(int) * N);

    // initialization of the host arrays
    for (int i=0; i<N; i++) {
        host_A[i] = rand() % 100;
        host_B[i] = rand() % 100;
        host_C[i] = rand() % 100;
    }

    // allocation of the GPU arrays
    cudaSetDevice(0); // set the working device
    int *dev_A;
    int *dev_B;
    int *dev_C;

    gpuErrchk(cudaMalloc((void**) &dev_A, N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_B, N*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_C, N*sizeof(int)));

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, N*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_B, host_B, N*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_C, host_C, N*sizeof(int), cudaMemcpyHostToDevice));

    // perform computation on GPU
    unsigned int numBlocks = std::ceil(((double) N) / BLOCK_DIM);
    kernel<<<numBlocks, BLOCK_DIM>>>(dev_A, dev_B, dev_C);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());

    // copy results from GPU memory to the host memory
    gpuErrchk(cudaMemcpy(host_check, dev_C, N*sizeof(int), cudaMemcpyDeviceToHost));

    // deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));
    gpuErrchk(cudaFree(dev_C));

    // check results
    for (int i=0; i<N; i++) {
        int tmp = (host_A[i] & host_B[i]) | host_C[i];
        if (tmp != host_check[i]) {
            std::cout << "Result error!" << std::endl;
            abort();            
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // ceallocate host memory
    free(host_A);
    free(host_B);
    free(host_C);
    free(host_check);

    return 0;
}
