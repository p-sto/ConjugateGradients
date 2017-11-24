/*Contains implementation of gpu_utils functions and structs.*/

extern "C" {
#include <cuda_runtime.h>
#include "gpu_utils.h"
}

GPU_data *get_gpu_devices_data(){
	/*Get information about CUDA devices on host.*/
	GPU_data *gpu_data;
	gpu_data = (GPU_data *)malloc(sizeof(GPU_data));
	gpu_data->devices_number = 0;
	cudaGetDeviceCount(&gpu_data->devices_number);
	if (gpu_data->devices_number != 0){
		gpu_data->devices = (GPU_device *)malloc(gpu_data->devices_number * sizeof(GPU_device));
		for (int i = 0; i < gpu_data->devices_number; i ++){
			cudaDeviceProp devProp;
			cudaGetDeviceProperties(&devProp, i);

			gpu_data->devices[i].name = devProp.name;
			gpu_data->devices[i].warp_size = devProp.warpSize;
			gpu_data->devices[i].max_threads_per_block = devProp.maxThreadsPerBlock;
			gpu_data->devices[i].clock_rate = devProp.clockRate;
			gpu_data->devices[i].multiprocessors_number = devProp.multiProcessorCount;
			gpu_data->devices[i].kernel_execution_timeout = (devProp.kernelExecTimeoutEnabled ? "Yes" : "No");
			gpu_data->devices[i].total_global_memory = devProp.totalGlobalMem;
			gpu_data->devices[i].total_shared_memory_per_block = devProp.sharedMemPerBlock;
		}
	}
	return gpu_data;
}