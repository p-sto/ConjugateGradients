/* Implementation of Preconditioned Conjugate Gradient method solver.
 *
 * Author: Pawel Stoworowicz
 * Contact: stoworow@gmail.com *
 * */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "pcg_solver.h"
#include "mkl.h"

#define FREE_STACK mkl_free(Ax);\
	mkl_free(q);\
	mkl_free(x);\
	mkl_free(d);\
	mkl_free(s);\
	mkl_free(diag);\
	mkl_free(inv_diagonal);\
	free(pc_config)


double *getInvertedDiagonal(Matrix *matrix, double *diag){
	/*Perform diagonal inversion*/
	double *inv_diagonal = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	for(int i = 0; i < matrix->size; i++){
		inv_diagonal[i] = 1/diag[i];
	}
	return inv_diagonal;
}

double *getDiagonal(Matrix *matrix){
	/*Return diagonal vector from CSR matrix.*/
	double *diag = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	int k = 0;
	for (int i = 0; i < matrix->size; i++){
		for (int j = matrix->I_column[i]; j < matrix->I_column[i + 1]; j++){
			if(matrix->J_row[j] == i){
				diag[k] = matrix->val[j];
				k++;
			}
		}
	}
	return diag;
}

double* perform_preconditioning(Matrix *matrix, double *vec, pc_cfg *pc_config){
	/*Performs preconditioning*/
	double *res = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	if 	(strcmp(pc_config->preconditioner, "jacobi") == 0){
		for (int i = 0; i < matrix->size; i++){
			res[i] = vec[i] * pc_config->inverted_diagonal[i];
		}
	} else {
		printf("No such preconditioner: %s, exiting", pc_config->preconditioner);
		exit(1);
	}
	return res;
}

int pcg_solver(Matrix *matrix, double *x_vec, double *rhs, double *res_vec, char* preconditioner){
	/* Conjugate gradient method solver implementation.
	 * Inputs: Matrix, x,b and res_vector
	 * Returns: number of iterations
	 * */

	int k, max_iter;
	double beta, tol, alpha, dTq, dot_zero, dot_new, dot_old;
	double *Ax, *x, *d, *q, *s, *diag, *inv_diagonal;
	k = 0;
	beta = 0.0;
	tol = 1e-2f;
	max_iter = 200;

	d = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	x = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	q = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	s  = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	Ax = (double *)mkl_malloc(matrix->size * sizeof(double), 64);

	diag = getDiagonal(matrix);
	inv_diagonal = getInvertedDiagonal(matrix, diag);

	pc_cfg *pc_config = (pc_cfg *)malloc(sizeof(pc_config));
	pc_config->preconditioner = preconditioner;
	pc_config->diagonal = diag;
	pc_config->inverted_diagonal = inv_diagonal;

	/*Calculate Ax matrix*/
	mkl_cspblas_dcsrgemv("N", &(matrix->size), matrix->val, matrix->I_column, matrix->J_row, x_vec, Ax);
	/*Calculate rhs=rhs-Ax matrix*/
	cblas_daxpy(matrix->size, -1.0, Ax, 1 , rhs, 1);
	/*CG: Copy updated rhs (residuum) to d vector*/
	cblas_dcopy(matrix->size, rhs, 1, d, 1);
	/*Preconditioning part*/
	d = perform_preconditioning(matrix, rhs, pc_config);
	/*CG: calculate dot r'*d, assign it to dot_new */
	dot_new = cblas_ddot(matrix->size, rhs, 1 ,d, 1);
	/*assign dot_new to dot_zero*/
	dot_zero = dot_new;
	while (dot_new >  tol * tol * dot_zero && k < max_iter) {
		/*Calculate q=A*d vector*/
		mkl_cspblas_dcsrgemv("N", &(matrix->size), matrix->val, matrix->I_column, matrix->J_row, d, q);
		/*Calculate alpha:*/
		dTq = cblas_ddot(matrix->size, d, 1, q, 1);
		alpha = dot_new / dTq;
		/*Calculate x=x+alpha*d*/
		cblas_daxpy(matrix->size, alpha, d, 1, x, 1);
		/*Calculate r=r-alpha*q*/
		cblas_daxpy(matrix->size, (-1) * alpha, q, 1, rhs, 1);
		/*Preconditioning part*/
		s = perform_preconditioning(matrix, rhs, pc_config);
		/*Assign dot_old = dot_new*/
		dot_old = dot_new;
		/*CG:Assign dot_new = r'*s*/
		dot_new = cblas_ddot(matrix->size, rhs, 1, s, 1);
		beta = dot_new / dot_old;
		/*Scale beta*d*/
		cblas_dscal(matrix->size, beta, d, 1);
		/*CG:Calculate d=s+beta*d*/
		cblas_daxpy(matrix->size, 1.0, s, 1, d, 1);
		k++;
	}
	FREE_STACK;
	return k;
}