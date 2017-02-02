#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel, Each thread takes care of one element of c
__global__ void vecAdd(double *a, double *b, double *c, int n)
{
	// Get global thread ID
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// Make sure not to go out of bounds
	if (idx < n)
		c[idx] = a[idx] + b[idx];
}

int main (int argc, char **argv)
{
	// Size of vectors for addition
	int n = 10000000;

	// Host vectors
	double *h_a, *h_b, *h_c, *h_h;

	// Device vectors
	double *d_a, *d_b, *d_c;

	// Size in bytes for each vector
	size_t bytes = n*sizeof(double);

	// Allocate memory for each vector on host
	h_a = (double *)malloc(bytes);
	h_b = (double *)malloc(bytes);
	h_c = (double *)malloc(bytes);
	h_h = (double *)malloc(bytes);

	// Allocate memory for each vector on GPU
	cudaMalloc((void **)&d_a, bytes);
	cudaMalloc((void **)&d_b, bytes);
	cudaMalloc((void **)&d_c, bytes);	

	// Initialize all vectors on host
	int i;
	for (i = 0 ; i < n ; i++)
	{
		h_a[i] = sin(i) * sin(i);
		h_b[i] = cos(i) * cos(i);
	}

	// Copy host vectors to device
	cudaMemcpy( d_a, h_a, bytes, cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, bytes, cudaMemcpyHostToDevice );
	
	int blockSize, gridSize;

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

	// Release device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Release host memory
	free(h_a);
	free(h_b);
	free(h_c);
	free(h_h);

	return 0;
}
