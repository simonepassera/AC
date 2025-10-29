/* CUDA application computing the sum of two integer arrays */

#include<iostream>
#include<stdlib.h>
#include<stdint.h>

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

__global__ void vector_sum(int *A, int *B, int *C, int L)
{
    int threadUID = threadIdx.x + blockIdx.x * blockDim.x;
    if (threadUID < L) {
        C[threadUID] = A[threadUID] + B[threadUID];
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr<< "Usage: " << argv[0] << " <size>" << std::endl;
        exit(1);
    }
    int L = atoi(argv[1]);

    // allocation of the host arrays host_A and host_B
    int *host_A = (int *) malloc(sizeof(int) * L);
    int *host_B = (int *) malloc(sizeof(int) * L);
    int *host_C = (int *) malloc(sizeof(int) * L);

    // initialization of the host arrays
    for (int i=0; i<L; i++) {
        host_A[i] = rand() % 100;
        host_B[i] = rand() % 100;
    }

    // allocate GPU arrays
    cudaSetDevice(0); // set the working device
    int *dev_A, *dev_B, *dev_C;
    gpuErrchk(cudaMalloc((void**) &dev_A, L*sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_B, L *sizeof(int)));
    gpuErrchk(cudaMalloc((void**) &dev_C, L *sizeof(int)));

    uint64_t initial_time = current_time_nsecs();

    // copy data to GPU memory
    gpuErrchk(cudaMemcpy(dev_A, host_A, L *sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dev_B, host_B, L *sizeof(int), cudaMemcpyHostToDevice));

    // perform computation on GPU
    unsigned int numThreadsPerBlock = 512;
    unsigned int numBlocks = std::ceil(((double) L) / numThreadsPerBlock);

    vector_sum<<<numBlocks, numThreadsPerBlock>>>(dev_A, dev_B, dev_C, L);
    gpuErrchk(cudaPeekAtLastError());

    // copy results from GPU memory
    gpuErrchk(cudaMemcpy(host_C, dev_C, L*sizeof(int), cudaMemcpyDeviceToHost));

    uint64_t end_time = current_time_nsecs();
    uint64_t elapsed = end_time - initial_time;
    std::cout << "Elapsed time: " << ((float) elapsed/1000.0) << " usec" << std::endl;

    // deallocate GPU memory
    gpuErrchk(cudaFree(dev_A));
    gpuErrchk(cudaFree(dev_B));
    gpuErrchk(cudaFree(dev_C));

    // check results
    int *check = (int *) malloc(sizeof(int) * L);
    for (int i=0; i<L; i++) {
        check[i] = host_A[i] + host_B[i];
        if (check[i] != host_C[i]) {
            std::cerr << "Error result" << std::endl;
            exit(1);
        }
    }
    std::cout << "Result is ok!" << std::endl;

    // deallocate host memory
    free(host_A);
    free(host_B);
    free(host_C);
    free(check);

    return 0;
}
