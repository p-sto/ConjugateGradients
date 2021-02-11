/*Contains implementation for gpu_cg_solver functions.*/

#include <stdio.h>
#include <stdint.h>
#include "ckernels.h"
#include "mkl.h"

extern "C"
{
#include "../misc/utils.h"
#include "gpu_utils.h"
#include "cg_solver_gpu.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "../misc/logger.h"
#include "../misc/math_functions.h"
}

#define threadsPerBlock 128
#define CHECK_FOR_STATUS(status) logger(LEVEL_DEBUG, "cublas status = %s", cublasGetErrorString(status))

#define COPY_TO_DEVICE(dest, source, size) cudaMemcpy(dest, source, size, cudaMemcpyHostToDevice)
#define COPY_TO_HOST(dest, source, size) cudaMemcpy(dest, source, size, cudaMemcpyDeviceToHost)

#define FREE_DEVICE_STACK \
	cudaFree(d_r);\
	cudaFree(d_helper);\
	cudaFree(d_x);\
	cudaFree(d_rhs);\
	cudaFree(d_d);\
	cudaFree(d_Ax);\
	cudaFree(d_q);\
	cudaFree(d_val);\
	cudaFree(d_I);\
	cudaFree(d_J);\
	cudaFree(d_beta);\
	cudaFree(d_alfa);\
	cudaFree(d_alpha_zero);\
	cudaFree(d_dot_new);\
	cudaFree(d_norm);\
	cudaFree(d_dot_zero);\
	cudaFree(d_dot_old);\
	cudaFree(d_dTq);


int gpu_conjugate_gradient_solver(MatrixInt *matrix, double *x_vec, double *rhs, double *res_vec, GPU_data *gpu_data){
	/*Single GPU CG solver using (mostly) cublas and custom kernels*/
    int *d_I = NULL, *d_J = NULL;
    int k, max_iter;
	double *d_beta, *d_alpha_zero, *d_alfa;
	double *d_Ax, *d_x, *d_d, *d_q, *d_rhs, *d_r, *d_helper, *d_norm, *d_dot_new, *d_dot_zero, *d_dot_old, *d_dTq, *d_val;
    const double tol = 1e-2f;

	double h_dot = 0;
	double h_dot_zero = 0;
	double alpha = 1.0;
	double alpham1 = -1.0;
	double beta = 0.0;

    d_alpha_zero = &alpham1;

    k = 0;
	max_iter = 200;

	size_t size = matrix->size * sizeof(double);
	size_t d_size = sizeof(double);

	cusparseHandle_t cusparseHandle = 0;
	cusparseCreate(&cusparseHandle);

	cusparseMatDescr_t descr = 0;
	cusparseCreateMatDescr(&descr);

	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	logger(LEVEL_DEBUG, "%s", "Mallocing CUDA device memory");
	cudaMalloc((double **)&d_r, size);
	cudaMalloc((double **)&d_helper, size);
	cudaMalloc((double **)&d_x, size);
	cudaMalloc((double **)&d_rhs,	size);
	cudaMalloc((double **)&d_d, size);
	cudaMalloc((double **)&d_Ax, size);
	cudaMalloc((double **)&d_q, size);
    cudaMalloc((double **)&d_alpha_zero, size);

	cudaMalloc((double **)&d_val, matrix->non_zero * sizeof(double));
	cudaMalloc((int **)&d_J, matrix->non_zero * sizeof(int));
	cudaMalloc((int **)&d_I, (matrix->size + 1) * sizeof(int));

    cudaMalloc((double **)&d_dTq, d_size);
    cudaMalloc((double **)&d_beta, d_size);
	cudaMalloc((double **)&d_alfa, d_size);
	cudaMalloc((double **)&d_dot_new, d_size);
	cudaMalloc((double **)&d_dot_zero, d_size);
	cudaMalloc((double **)&d_norm, d_size);

    logger(LEVEL_DEBUG, "%s", "Copying to device.");
	COPY_TO_DEVICE(d_val, matrix->val, matrix->non_zero * sizeof(double));
	COPY_TO_DEVICE(d_J, matrix->J_row, matrix->non_zero * sizeof(int));
	COPY_TO_DEVICE(d_I, matrix->I_column, (matrix->size + 1) * sizeof(int));
	COPY_TO_DEVICE(d_x, x_vec, matrix->size * sizeof(double));
	COPY_TO_DEVICE(d_rhs, rhs, matrix->size * sizeof(double));

	int blocksPerGrid = ((matrix->size + threadsPerBlock - 1) / threadsPerBlock );
	while (blocksPerGrid % threadsPerBlock != 0){
		blocksPerGrid++;
	}

    /*Calculate Ax matrix*/
	cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                   matrix->size, matrix->size, matrix->non_zero,  &alpha, descr, d_val, d_I, d_J, d_x, &beta, d_Ax);

    /*Calculate rhs=rhs-Ax matrix*/
    cudaDaxpy<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_alpha_zero, d_Ax, d_rhs);

    /*CG: Copy updated rhs (residuum) to d vector*/
    copy_device_vectors<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_d, d_rhs);

    /*CG: calculate dot r'*r, assign it to dot_new */
	cudaDElementWiseMult<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_rhs, d_rhs, d_helper);
	cudaDvect_sum<<<blocksPerGrid, threadsPerBlock>>>(d_helper, d_dot_new, blocksPerGrid);

    /*assign dot_new to dot_zero*/
    copy_device_variables<<<1, gpu_data->devices[0].warp_size>>>(d_dot_zero, d_dot_new);

    COPY_TO_HOST(&h_dot, d_dot_new, sizeof(double));
    /*assign dot_new to dot_zero*/
    h_dot_zero = h_dot;
    printf("tol * tol * h_dot_zero = %f\n", tol * tol * h_dot_zero);
    printf("h_dot = %f\n", h_dot);
	while (h_dot >  tol * tol * h_dot_zero && k < max_iter) {
		/*Calculate q=A*d vector*/
		cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, matrix->size, matrix->size, matrix->non_zero,
						  &alpha, descr, d_val, d_I, d_J, d_d, &beta, d_q);

		/*Calculate dTq = d'*q*/
        cudaDElementWiseMult<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_d, d_q, d_helper);
		/*Calculate alpha = dot_new/dTq*/
        cudaDvect_sum<<<blocksPerGrid, threadsPerBlock>>>(d_helper, d_dTq, blocksPerGrid);
        cudaDdiv_scalar<<<1, gpu_data->devices[0].warp_size>>>(d_alfa, d_dot_new, d_dTq);

        double tmp;
        double *p_tmp;
        p_tmp = &tmp;
		COPY_TO_HOST(p_tmp, d_dot_new, sizeof(double));
		logger(LEVEL_DEBUG, "d_dot_new val = %f", tmp);
		COPY_TO_HOST(p_tmp, d_dTq, sizeof(double));
		logger(LEVEL_DEBUG, "d_dTq val = %f", tmp);
		COPY_TO_HOST(p_tmp, d_alfa, sizeof(double));
		logger(LEVEL_DEBUG, "d_alfa val = %f", tmp);

		/*Calculate x=x+alpha*d*/
		cudaDaxpy<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_alfa, d_d, d_x);

		/*Calculate r=r-alpha*q*/
		cudaDaxpy<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_alpha_zero, d_q, d_rhs);
		/*Assign dot_old = dot_new*/
        copy_device_variables<<<1, gpu_data->devices[0].warp_size>>>(d_dot_old, d_dot_new);

		/*Calculate dot_new = r'*r*/
        cudaDElementWiseMult<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_rhs, d_rhs, d_helper);
        cudaDvect_sum<<<blocksPerGrid, threadsPerBlock>>>(d_helper, d_dot_new, blocksPerGrid);

        /*Calculate beta = d_new/d_old */
        cudaDdiv_scalar<<<1, gpu_data->devices[0].warp_size>>>(d_beta, d_dot_new, d_dot_old);

		/*CG:Calculate d=r+beta*d*/
		cudaDaxpyz<<<blocksPerGrid, threadsPerBlock>>>(matrix->size, d_beta, d_d, d_d, d_r);
        COPY_TO_HOST(&h_dot, d_dot_new, sizeof(double));
        cudaThreadSynchronize();
		k++;
	}
	cusparseDestroy(cusparseHandle);
	cudaDeviceReset();
	FREE_DEVICE_STACK
	return k;
}
