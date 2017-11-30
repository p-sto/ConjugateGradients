/*Contains implementations of miscellaneous functions.*/

#ifdef OS_WINDOWS
	#include <time.h>
#else
	#include <sys/time.h>
#endif

#include <string.h>
#include <stdio.h>
#include "cg_solver.h"
#include "pcg_solver.h"
#include "utils.h"
#include "mkl.h"
#include "gpu_utils.h"
#include "cg_colver_gpu.h"
#include <cuda_runtime.h>

double get_time(){
	/* Return ``double`` time stamp for execution time measurements. Implementation depends on OS. */
#ifdef OS_WINDOWS
	double t = 0.0;
	clock_t start;
	start = clock();
	t = ((double)start)/CLOCKS_PER_SEC;
	return t;
#else
	struct timeval tp;
	double sec, usec;
	gettimeofday(&tp, 0);
	sec = (double)(tp.tv_sec);
	usec = (double)(tp.tv_usec)/1E6;
	return sec + usec;
#endif
}

Matrix* getMatrixCRS(InputConfig *input_cfg){
	/* Load Matrix stored in CSR format from file.*/
	char *fil_name;
	fil_name = input_cfg->filename;
	if (fil_name == NULL)
		return NULL;
	FILE *pf;
	pf = fopen(fil_name, "r");
	if(pf == NULL){
		printf("Can't open the file: %s", fil_name);
	}

	double v;
	int c, non_zero, inx;
	int matsize, max_rows;

	// Load first line of file with information about size and non zero element in matrix
	fscanf(pf,"%d %d %d", &matsize, &non_zero, &max_rows);

	Matrix *matrix = (Matrix *)malloc(sizeof(Matrix));
	matrix->size = matsize;
	matrix->non_zero = non_zero;
	matrix->I_column = (MKL_INT *)mkl_malloc((matsize + 1) * sizeof(MKL_INT), 64);
	matrix->J_row = (MKL_INT *)mkl_malloc(non_zero * sizeof(MKL_INT), 64 );
	matrix->val = (double *)mkl_malloc(non_zero * sizeof(double), 64);

	int n = 0;
	while( ! feof(pf)){
		if(n < matsize + 1){
			fscanf(pf, "%le %d %d", &v, &c, &inx);
			matrix->val[n] = v;
			matrix->J_row[n] = c;
			matrix->I_column[n] = inx;
		}
		else{
			fscanf(pf, "%le %d", &v, &c);
			matrix->val[n] = v;
			matrix->J_row[n] = c;
		}
		n++;
	}
	fclose(pf);
	pf = NULL;
	return matrix;
}

InputConfig* arg_parser(int argc, char **argv){
	/*Parse argument line*/
	InputConfig *input_cfg = (InputConfig *)malloc(sizeof(InputConfig));
	input_cfg->num_of_threads = 1;
	input_cfg->gpu = 0;
	input_cfg->filename = NULL;
	input_cfg->preconditioner = NULL;

	int mt = 0;
	int fil_flag = 0;
	for (int i = 0; i < argc; i++){
		if (mt)
			input_cfg->num_of_threads = atoi(argv[i]);
			mt = 0;
		if (strcmp(argv[i], "-mt") == 0)
			mt = 1;
		if (fil_flag)
			input_cfg->filename = argv[i];
			fil_flag = 0;
		if (strcmp(argv[i], "-i") == 0)
			fil_flag = 1;
		if (strcmp(argv[i], "--pcg_jacobi") == 0)
			input_cfg->preconditioner = "jacobi";
		if (strcmp(argv[i], "--gpu") == 0)
			input_cfg->gpu = 1;
	}
	return input_cfg;
}

int launch_solver(Matrix *matrix, double *x_vec, double *b_vec, double *res_vec, InputConfig *input_cfg){
	/*Launch proper solver based on input config data.*/
	GPU_data *gpu_data;
	gpu_data = get_gpu_devices_data();

	int total_iter = 0;
	if (gpu_data->devices_number > 0 && input_cfg->gpu == 1){
		printf("Launching GPU CG.\n");
		printf("Number of CUDA devices: %d\n", gpu_data->devices_number);
		cudaDeviceReset();
		total_iter = gpu_conjugate_gradient_solver(matrix, x_vec, b_vec, res_vec, gpu_data);
	} else {
		if (input_cfg->preconditioner == NULL){
			printf("Launching CPU CG.\n");
			total_iter = conjugate_gradient_solver(matrix, x_vec, b_vec, res_vec);
		} else {
			printf("Launching CPU PCG with %s preconditioner.\n", input_cfg->preconditioner);
			total_iter = pcg_solver(matrix, x_vec, b_vec, res_vec, input_cfg->preconditioner);
		}
	}
	free(gpu_data);
	return total_iter;
}
