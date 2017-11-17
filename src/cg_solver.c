/* Implementation of Conjugate Gradient method solver.
 *
 * Author: Pawel Stoworowicz
 * Contact: stoworow@gmail.com *
 * */

#include <stdio.h>
#include <stdlib.h>
#include "cg_solver.h"
#include "mkl.h"

#define FREE_STACK \
	mkl_free(Ax);\
	mkl_free(q);\
	mkl_free(x);\
	mkl_free(d)

int conjugate_gradient_solver(Matrix *matrix, double *x_vec, double *rhs, double *res_vec){
	/* Conjugate gradient method solver implementation.
	 * Inputs: Matrix, x,b and res_vector
	 * Returns: number of iterations
	 * */

	int k, max_iter;
	double beta, tol, alpha, dTq, dot_zero, dot_new, dot_old;
	double *Ax, *x, *d, *q;
	k = 0;
	beta = 0.0;
	tol = 1e-2f;
	max_iter = 200;

	d = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	x = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	q = (double *)mkl_malloc(matrix->size * sizeof(double), 64);
	Ax = (double *)mkl_malloc(matrix->size * sizeof(double), 64);

	/*Calculate Ax matrix*/
	mkl_cspblas_dcsrgemv("N", &(matrix->size), matrix->val, matrix->I_column, matrix->J_row, x_vec, Ax);
	/*Calculate rhs=rhs-Ax matrix*/
	cblas_daxpy(matrix->size, -1.0, Ax, 1 , rhs, 1);
	/*CG: Copy updated rhs (residuum) to d vector*/
	cblas_dcopy(matrix->size, rhs, 1, d, 1);
	/*CG: calculate dot r'*r, assign it to dot_new */
	dot_new = cblas_ddot(matrix->size, rhs, 1 ,rhs, 1);
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
		/*Assign dot_old = dot_new*/
		dot_old = dot_new;
		/*CG:Assign dot_new = r'*r*/
		dot_new = cblas_ddot(matrix->size, rhs, 1, rhs, 1);
		beta = dot_new / dot_old;
		/*Scale beta*d*/
		cblas_dscal(matrix->size, beta, d, 1);
		/*CG:Calculate d=r+beta*d*/
		cblas_daxpy(matrix->size, 1.0, rhs, 1, d, 1);
		k++;
	}
	FREE_STACK;
	return k;
}