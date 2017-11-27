/*Contains implementation of custom kernels for CUDA devices.*/

#include "ckernels.h"

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
