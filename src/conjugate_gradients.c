/* File contains implementation of Conjugate Gradient method using Intel MKL library and nVidia CUDA.
 *
 * Author: Pawel Stoworowicz
 *
 * */

#define FREE_STACK_COMMON \
    free(matrix);\
    free(input_cfg);\

#define FREE_STACK_ALL \
    FREE_STACK_COMMON\
    free(res_vec);\
    free(b_vec);\
    free(x_vec);

//mkl_free(res_vec);\
	mkl_free(b_vec);\
	mkl_free(x_vec);


#include <stdlib.h>
#include <omp.h>
#include "stdio.h"
#include "misc/utils.h"
#include "mkl.h"
#include "misc/logger.h"


int main(int argc, char **argv) {
    /*Main function - thanks captain obvious!*/
    logger_clean();
    logger(LEVEL_INFO, "%s", "Conjugate Gradient solver using MKL.");
    InputConfig *input_cfg;
    input_cfg = arg_parser(argc, argv);
    logger(LEVEL_INFO, "Num of threads: %d", input_cfg->num_of_threads);
    omp_set_num_threads(input_cfg->num_of_threads);
    MatrixInt *matrix;
    matrix = getMatrixCRSInt(input_cfg);
    if (matrix == NULL) {
        logger(LEVEL_ERROR, "%s", "ERROR - could not read matrix, exiting.");
        FREE_STACK_COMMON;
        exit(1);
    }
    logger(LEVEL_DEBUG, "%s", "Matrix parameters:");
    logger(LEVEL_DEBUG, "Size = %d, non-zero elements = %d", (int) matrix->size, (int) matrix->non_zero);

    double *x_vec, *b_vec, *res_vec;
    double t_start, t_stop;
    /*
    x_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
    b_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
    res_vec = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
    */

    x_vec = (double *) malloc(matrix->size * sizeof(double));
    b_vec = (double *) malloc(matrix->size * sizeof(double));
    res_vec = (double *) malloc(matrix->size * sizeof(double));

    int total_iter = 0;
    for (int i = 0; i < matrix->size; i++) {
        x_vec[i] = 1.0;
        b_vec[i] = 0.0;
        res_vec[i] = 0.0;
    }
    t_start = get_time();
    total_iter = launch_solver(matrix, x_vec, b_vec, res_vec, input_cfg);
    t_stop = get_time();
    logger(LEVEL_INFO, "Conjugate gradient method finished within %.3f [secs] in total %d iterations.", t_stop - t_start, total_iter);

    FREE_STACK_ALL;
    exit(0);
}
