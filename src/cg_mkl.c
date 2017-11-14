/* File contains implementation of Conjugate Gradient method using Intel MKL library.
 *
 * Author: Pawel Stoworowicz
 * Contact: stoworow@gmail.com
 *
 * */

#define FREE_STACK free(matrix);\
	free(input_cfg)

#include <stdlib.h>
#include "stdio.h"
#include "utils.h"
#include "cg_solver.h"
#include "mkl.h"


int main(int argc, char** argv){
	printf("Conjugate Gradient solver using MKL.\n");
	InputConfig *input_cfg;
	input_cfg = arg_parser(argc, argv);
	printf("Num of threads: %d\n", input_cfg->num_of_threads);
	Matrix *matrix;
	matrix = getMatrixCRS(input_cfg);
	if (matrix == NULL){
		printf("ERROR - could not read matrix, exiting.\n");
		FREE_STACK;
		exit(1);
	}
	printf("Matrix parameters:\n");
	printf("Size = %d, non-zero elements = %d\n", matrix->size, matrix->non_zero);

	// Allocate x, b and res_vec vectors and time stamps:
	double *x_vec, *b_vec, *res_vec;
	double t_start, t_stop;

	x_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	b_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	res_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	int total_iter = 0;

	t_start = get_time();
	total_iter = conjugate_gradient_solver(matrix, x_vec, b_vec, res_vec);
	t_stop = get_time();
	printf("Conjugate gradient method finished within %.3f [secs] in total %d iterations.\n", t_stop - t_start, total_iter);

	FREE_STACK;
}
