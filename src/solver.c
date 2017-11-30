/* File contains implementation of Conjugate Gradient method using Intel MKL library.
 *
 * Author: Pawel Stoworowicz
 * Contact: stoworow@gmail.com
 *
 * */

#define FREE_STACK_COMMON \
	free(matrix);\
	free(input_cfg);\

#define FREE_STACK_ALL \
	FREE_STACK_COMMON\
	mkl_free(res_vec);\
	mkl_free(b_vec);\
	mkl_free(x_vec)

#include <stdlib.h>
#include <omp.h>
#include "stdio.h"
#include "utils.h"
#include "mkl.h"


int main(int argc, char** argv){
	/*Main function - thanks captain obvious!*/
	printf("Conjugate Gradient solver using MKL.\n");
	InputConfig *input_cfg;
	input_cfg = arg_parser(argc, argv);
	printf("Num of threads: %d\n", input_cfg->num_of_threads);
	omp_set_num_threads(input_cfg->num_of_threads);
	Matrix *matrix;
	matrix = getMatrixCRS(input_cfg);
	if (matrix == NULL){
		printf("ERROR - could not read matrix, exiting.\n");
		FREE_STACK_COMMON;
		exit(1);
	}
	printf("Matrix parameters:\n");
	printf("Size = %d, non-zero elements = %d\n", (int)matrix->size, (int)matrix->non_zero);

	double *x_vec, *b_vec, *res_vec;
	double t_start, t_stop;
	x_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	b_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	res_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	int total_iter = 0;
	for (int i = 0; i < matrix->size; i++) {
		x_vec[i] = 1;
		b_vec[i] = 0;
		res_vec[i] = 0;
	}
	t_start = get_time();
	total_iter = launch_solver(matrix, x_vec, b_vec, res_vec, input_cfg);
	t_stop = get_time();
	printf("Conjugate gradient method finished within %.3f [secs] in total %d iterations.\n", t_stop - t_start, total_iter);

	FREE_STACK_ALL;
	exit(0);
}
