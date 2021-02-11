/*Contains implementation of custom kernels for CUDA devices.*/

#include "ckernels.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define threadsPerBlock 128

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

__global__ void cudaDdiv_scalar(double *res, double *divided, double *divider){
	/*Division of scalar elements on a single CUDA thread*/
	if (threadIdx.x == 0) {
		res[0] = divided[0] / divider[0];
	}
}

__global__ void cudaDaxpy(int num_elements, double *alpha, double *x, double *y){
	/*Perform computations of AXPY operations: y[i] = y[i] + alpha * x[i]*/
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_elements) {
		y[i] = y[i] + alpha[0] * x[i];
	}
}

__global__ void cudaDaxpyz(int num_elements, double *alpha, double *y, double *x, double *z){
	/*Compute y[i] = alpha * x[i] + z[i] */
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < num_elements) {
		y[i] = alpha[0] * x[i] + z[i];
	}
}

__global__ void copy_device_variables(double *destination, double *source){

    if (threadIdx.x==0){
        *destination = *source;
    }
}

__global__ void copy_device_vectors(int numElements, double *destination, double *source){

	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		destination[i] = source[i];
	}
}

__global__ void cudaDElementWiseMult(int numElements, double *x, double *y, double *res){

	__shared__ double temp[threadsPerBlock];

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int tid = threadIdx.x;
	temp[tid] = 0;
	res[i] = 0;
	int blockID = blockIdx.x;
	__syncthreads();

	if (i < numElements){
		temp[tid]= x[i] * y[i];
	}
	__syncthreads();

	for (int s=blockDim.x/2; s>32; s>>=1){
		if (tid<s){
			temp[tid] += temp[tid+s];
		}
		__syncthreads();
	}
	if (tid<32){
		temp[tid] += temp[tid+32];__syncthreads();
		temp[tid] += temp[tid+16];__syncthreads();
		temp[tid] += temp[tid+8];__syncthreads();
		temp[tid] += temp[tid+4];__syncthreads();
		temp[tid] += temp[tid+2];__syncthreads();
		temp[tid] += temp[tid+1];__syncthreads();
	}

	__syncthreads();
	if(tid == 0){
		res[blockID] = temp[0];
	}
}

__global__ void cudaDvect_sum(double *input, double *res, int blocksPerGrid){

	__shared__ double temp2[threadsPerBlock];
	__shared__ double temp3[2];
	int tid = threadIdx.x;
	temp3[0] = 0;
	*res = 0;
	temp2[tid] = 0;
	__syncthreads();

	for(int j = 0; j<(int)(blocksPerGrid/threadsPerBlock); j++){
		temp2[tid] += input[tid*(int)floorf(blocksPerGrid/threadsPerBlock)+j];
	}
	__syncthreads();

	if (tid == 0){
		for (int z = 0; z< threadsPerBlock/2 ; z++){
			temp3[0] += temp2[z] + temp2[z+threadsPerBlock/2];
		}
		res[0] = temp3[0];
	}
}