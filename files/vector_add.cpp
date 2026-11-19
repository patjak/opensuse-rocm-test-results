#include <hip/hip_runtime.h>
#include <iostream>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    int N = 100000;
    size_t bytes = N * sizeof(float);
    float *h_A = new float[N], *h_B = new float[N], *h_C = new float[N];
    for(int i=0; i<N; i++) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            std::cerr << "HIP error: " << hipGetErrorName(err) << ": " << hipGetErrorString(err) << std::endl; \
            std::cout << "HIP_TEST_FAILED" << std::endl; \
            delete[] h_A; delete[] h_B; delete[] h_C; \
            if (d_A) hipFree(d_A); if (d_B) hipFree(d_B); if (d_C) hipFree(d_C); \
            return 1; \
        } \
    } while (0)

    CHECK_HIP(hipMalloc(&d_A, bytes));
    CHECK_HIP(hipMalloc(&d_B, bytes));
    CHECK_HIP(hipMalloc(&d_C, bytes));

    CHECK_HIP(hipMemcpy(d_A, h_A, bytes, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B, h_B, bytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(vectorAdd, dim3((N+255)/256), dim3(256), 0, 0, d_A, d_B, d_C, N);
    CHECK_HIP(hipGetLastError());
    CHECK_HIP(hipDeviceSynchronize());

    CHECK_HIP(hipMemcpy(h_C, d_C, bytes, hipMemcpyDeviceToHost));

    bool pass = true;
    for(int i=0; i<N; i++) {
        if(h_C[i] != 3.0f) { pass = false; break; }
    }
    std::cout << (pass ? "HIP_TEST_SUCCESS" : "HIP_TEST_FAILED") << std::endl;
    
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;
    return pass ? 0 : 1;
}
