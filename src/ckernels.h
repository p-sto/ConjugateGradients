/*Contains prototypes of custom kernels for CUDA devices.*/

__global__ void sDdiv(double *res, double *divided, double *divider);
__global__ void axpy(int num_elements, double alpha, double *x, double *y);
