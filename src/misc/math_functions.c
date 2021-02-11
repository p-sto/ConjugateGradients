/*Contains implementations of custom math functions*/

#include <math.h>
#include <omp.h>
#include "math_functions.h"

double euclidean_norm(int size, double *vector){
    double res = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < size; i++){
        res += pow(vector[i], 2);
    }
    return sqrt(res);
}
