/*Contains prototypes of gpu_utils functions and structs.*/

#ifndef GPU_UTILS_H
#define GPU_UTILS_H
typedef struct {
	char *name;
	int warp_size;
	int max_threads_per_block;
	int clock_rate;
	int multiprocessors_number;
	const char *kernel_execution_timeout;
	unsigned int total_global_memory;
	unsigned int total_shared_memory_per_block;
} GPU_device;

typedef struct {
	int devices_number;
	GPU_device *devices;
} GPU_data;

GPU_data *get_gpu_devices_data();

#endif