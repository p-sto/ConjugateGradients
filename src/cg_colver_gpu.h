/*Contains prototypes for gpu_cg_solver functions.*/

#ifndef CG_SOLVER_GPU_H
#define CG_SOLVER_GPU_H
#include "utils.h"
#include "gpu_utils.h"

int gpu_conjugate_gradient_solver(Matrix *matrix, double *x_vec, double *rhs, double *res_vec, GPU_data *gpu_data);

#endif