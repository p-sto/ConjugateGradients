/*Contains prototypes for miscellaneous functions.*/

#ifndef UTILS_H
#define UTILS_H
double get_time(void);

typedef struct {
	int size;
	int non_zero;
	int *I_column;
	int *J_row;
	double *val;
}Matrix;

typedef struct{
	char *filename;
	int num_of_threads;
} InputConfig;

Matrix* getMatrixCRS(InputConfig *input_cfg);

InputConfig* arg_parser(int argc, char **argv);
#endif