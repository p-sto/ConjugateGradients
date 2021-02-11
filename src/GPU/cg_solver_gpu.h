/*Contains prototypes for gpu_cg_solver functions.*/

#ifndef CG_SOLVER_GPU_H
#define CG_SOLVER_GPU_H
#include "../misc/utils.h"
#include "gpu_utils.h"

int gpu_conjugate_gradient_solver(MatrixInt *matrix, double *x_vec, double *rhs, double *res_vec, GPU_data *gpu_data);

#endif