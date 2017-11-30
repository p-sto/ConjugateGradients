/*Contains implementation of custom kernels for CUDA devices.*/

#include "ckernels.h"
#include <cublas_v2.h>

const char* cublasGetErrorString(cublasStatus_t status)
{
	switch(status)
	{
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "unknown error";
}

__global__ void sDdiv(double *res, double *divided, double *divider) {
	/*Division of scalar elements on a single CUDA thread*/
	if (threadIdx.x == 0) {
		res[0] = divided[0] / divider[0];
	}
}

__global__ void axpy(int num_elements, double alpha, double *x, double *y) {
	/*Perform computations of AXPY operations: y[i] = y[i] + alpha * x[i]*/
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_elements) {
		y[i] = y[i] + alpha * x[i];
	}
}

