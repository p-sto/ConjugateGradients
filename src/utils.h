/*Contains prototypes for miscellaneous functions.*/

#include "mkl.h"

#ifndef UTILS_H
#define UTILS_H
double get_time(void);

typedef struct {
	MKL_INT size;
	MKL_INT non_zero;
	MKL_INT *I_column;
	MKL_INT *J_row;
	double *val;
}Matrix;

typedef struct{
	char *filename;
	int num_of_threads;
	int gpu;
	char *preconditioner;
} InputConfig;

Matrix* getMatrixCRS(InputConfig *input_cfg);

InputConfig* arg_parser(int argc, char **argv);

int launch_solver(Matrix *matrix, double *x_vec, double *b_vec, double *res_vec, InputConfig *input_cfg);
#endif