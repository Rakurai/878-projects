#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "matrix.h"

// CUDA kernel, Each thread takes care of one element of c
__global__ void matrix_add(double *a, double *b, double *c, int n)
{
	// Get global thread ID
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure not to go out of bounds
	if (idx < n)
		c[idx] = a[idx] + b[idx];
}

int main(int argc, char **argv)
{
	int floatingpoint = atoi(argv[3]);

	void *input_matrix[2];
	int height[2], width[2];

	input_matrix[0] = read_matrix_file(argv[1], &height[0], &width[0], floatingpoint);
	input_matrix[1] = read_matrix_file(argv[2], &height[1], &width[1], floatingpoint);

	if (input_matrix[0] == NULL
	 || input_matrix[1] == NULL
	 || height[0] != height[1]
	 || width[0] != width[1]) {
		printf("Messed up input.\n");
		return 1;
	}

	// Size of vectors for addition
//	int n = 10000000;

	// Host vectors
//	double *h_a, *h_b, *h_c, *h_h;

	// Device vectors
	void *d_in1, *d_in2, *d_out;

	// Size in bytes for each vector
	size_t bytes = height[0]*width[0]*(floatingpoint ? sizeof(float) : sizeof(int));

	// Allocate memory for each vector on host
	void *output_matrix = malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc((void **)&d_in1, bytes);
	cudaMalloc((void **)&d_in2, bytes);
	cudaMalloc((void **)&d_out, bytes);	

	// Copy host vectors to device
	cudaMemcpy(d_a, input_matrix[0], bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, input_matrix[1], bytes, cudaMemcpyHostToDevice);
	
	int blockSize, gridSize;
/*
	// Number of threads in each thread block
	blockSize = 1024;

	// Number of thread blocks in grid
	gridSize = (int)ceil((float)n/blockSize);

	// Execute on CPU and GPU
	clock_t cpu_start, cpu_end;
	clock_t gpu_start, gpu_end;
	cpu_start = clock();
	for (i = 0 ; i < n ; i++)
	{
		h_h[i] = h_a[i] + h_b[i];
	}
	cpu_end = clock();
	gpu_start = clock();
	vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
	cudaDeviceSynchronize();
	gpu_end = clock();

	// Copy array back to host
	cudaMemcpy( h_c, d_c, bytes, cudaMemcpyDeviceToHost );

	// Sum up vector c and print result divided by n
	double sum = 0;
	for ( i = 0 ; i < n ; i++ )
		sum += h_c[i];
	printf("final:result: %f\n", sum/n);

	double cpu_time = ((double)cpu_end - cpu_start)/CLOCKS_PER_SEC;
	double gpu_time = ((double)gpu_end - gpu_start)/CLOCKS_PER_SEC;

	printf("CPU Time: %f\nGPU Time: %f\n", cpu_time, gpu_time);
*/
	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(input_matrix[0]);
	free(input_matrix[1]);
	free(output_matrix);

	return 0;
}
