/*Contains prototypes of custom kernels for CUDA devices.*/

#include <cublas_v2.h>
#include <cuda_runtime.h>

const char* cublasGetErrorString(cublasStatus_t status);
__global__ void cudaDdiv_scalar(double *res, double *divided, double *divider);
__global__ void cudaDaxpy(int num_elements, double *alpha, double *x, double *y);
__global__ void cudaDaxpyz(int num_elements, double *alpha, double *x, double *y, double *z);
__global__ void copy_device_variables(double *destination, double *source);
__global__ void copy_device_vectors(int numElements, double *destination, double *source);
__global__ void cudaDElementWiseMult(int numElements, double *x, double *y, double *res);
__global__ void cudaDvect_sum(double *input, double *res, int blocksPerGrid);
