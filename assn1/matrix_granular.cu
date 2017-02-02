/*This program implements the CUDA parallel version of matrix multiplication of two square matrices of equal size.
Shared Memory and thread granularity is used for optimizing performance.*/

#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#define TILE_WIDTH 8 /*Block Dimension of TILE_WIDTH x TILE_WIDTH*/
#define WIDTH 4096 /*WIdth of square Matrix*/
#define GRAN 2 /*Granularity - Number of blocks merged. Verified for powers of 2 i.e. 2, 4, 8, ...*/

void Matrix_Mul_gran(float *, float *, float *, int);

__global__ void Matrix_Mul_gran_Kernel(float* d_M, float* d_N, float* d_P, int Width)
{

int i, j, k;

/*allocate shared memory*/
__shared__ float ds_M[TILE_WIDTH*GRAN][TILE_WIDTH*GRAN];
__shared__ float ds_N[TILE_WIDTH*GRAN][TILE_WIDTH];

int bx, by, tx, ty, Row, Col;

/*Each thread evaluates multiple number of product elements depending on degree of granularity*/
float Pvalue[GRAN];

bx = blockIdx.x; by = blockIdx.y;
tx = threadIdx.x; ty = threadIdx.y;

Col = bx * TILE_WIDTH + tx;
Row = GRAN* by * TILE_WIDTH + ty;

for(i=0;i<GRAN;i++) Pvalue[i] = 0;
for (int m = 0; m < Width/TILE_WIDTH; m+=GRAN){
	/*Load shared memory*/
	for(i=0;i<GRAN;i++){
		ds_N[ty+i*TILE_WIDTH][tx] = d_N[Col+(m*TILE_WIDTH+ty+i*TILE_WIDTH)*Width];
		for(j=0;j<GRAN;j++){
			ds_M[ty+i*TILE_WIDTH][tx+j*TILE_WIDTH] = d_M[(Row+i*TILE_WIDTH)*Width + (m+j)*TILE_WIDTH+tx];
			}
		}
	__syncthreads();

	/*Evaluate product elements*/
	for(k=0;k<TILE_WIDTH*GRAN;k++){
		for(i=0;i<GRAN;i++){
			Pvalue[i] += ds_M[ty+i*TILE_WIDTH][k] * ds_N[k][tx];
			}
		}
	__syncthreads();
}

/*Write to device memory*/
for(i=0;i<GRAN;i++)
d_P[(Row+i*TILE_WIDTH)*Width+Col] = Pvalue[i];
}

int main(int argc, char** argv){

int W = WIDTH;
int i,j;
float *M, *N, *P;

struct timeval start, end;
long utime, seconds, useconds;    
	
M = (float*)malloc(W*W*sizeof(float));
N = (float*)malloc(W*W*sizeof(float));
P = (float*)malloc(W*W*sizeof(float));

/*Initialize the matrices to identity matrices. The result should be an identity matrix and thus can be verified easily*/
for(i =0;i<W;i++){
	for(j=0;j<W;j++){
		if(i==j){
			M[i*W+j]=1;
			N[i*W+j]=1;
			}
		else{
			M[i*W+j]=0;
			N[i*W+j]=0;
			}
	}
}

gettimeofday(&start, NULL);

/*Call function for matrix multiplication*/
Matrix_Mul_gran(M,N,P,W);
gettimeofday(&end, NULL);

seconds  = end.tv_sec  - start.tv_sec;
useconds = end.tv_usec - start.tv_usec;
utime = ((seconds) * 1000000 + useconds);

/*Print execution time in microseconds*/
printf("%d\t%ld\n", GRAN,utime);

free(M); free(N); free(P);

return 0;
}

/*Matrix multiplication function*/
void Matrix_Mul_gran(float *M, float *N, float *P, int W){
	int size = W*W*sizeof(float);
	float *Md, *Nd, *Pd;

	/*Initialize memory and copy data from host to device*/
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md,M,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd,N,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Pd, size);
	
	/*Launch Kernel*/
	/*Blocks along Grid Dimension 'Y' i.e. blockIdx.y are merged as per required granularity*/
	dim3 dimGrid(ceil(W/TILE_WIDTH), ceil(W/(TILE_WIDTH*GRAN)), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	Matrix_Mul_gran_Kernel<<<dimGrid, dimBlock>>>(Md, Nd, Pd, W);

	/*Copy result from device to host and free device memory*/
	cudaMemcpy(P,Pd,size,cudaMemcpyDeviceToHost);
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd); 
}
