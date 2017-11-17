/* Header file for Preconditioned Conjugate Gradient method */

#include "utils.h"

typedef struct {
	char *preconditioner;
	double *diagonal;
	double *inverted_diagonal;
}pc_cfg;

int pcg_solver(Matrix *matrix, double *x_vec, double *b_vec, double *res_vec, char* preconditioner);

double* perform_preconditioning(Matrix *matrix, double *vec, pc_cfg *pc_config);

double *getDiagonal(Matrix *matrix);

double *getInvertedDiagonal(Matrix *matrix, double *diag);