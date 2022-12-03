#include <stdio.h>
#include <stdlib.h>

#define N 32

__global__ void MatrixMultiply(float *d_A, float *d_B, float *d_C)
{
	
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int k=0; k<N; k++){
        d_C[row * N + col] += d_A[row * N + k] * d_B[k * N + col];	
    }
}

int main(){
	
	float h_A[N * N], h_B[N*N], h_C[N*N]; // host matrices

	// size to allocate
	size_t size = N*N*sizeof(float);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    //Initialize matrices on the host
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			h_A[i*N+j]=i;
			h_B[i*N+j]=i+1;
	    }
	}

	//Allocate Device memory
    float *d_A, *d_B, *d_C;	// devices matrices
	cudaMalloc(&d_A, size);
	cudaMalloc(&d_B, size);
	cudaMalloc(&d_C, size);

	//Allocate A and B to the Device
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//Invoke kernel
	dim3 blockPerGrid(1,1);
	dim3 threadPerBlock(N,N); // thread.x = N, thread.y = N

	cudaEventRecord(start, 0);	
	// Launch kernel
	MatrixMultiply<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C);
	cudaEventRecord(stop, 0);	
	//cudaEventSynchronize(stop);

	//Read C from device
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Elapsed time (ms): %f\n", milliseconds);
	
	// Check result on the CPU
	// For every row...
	for (int i = 0; i < N; i++) {
		// For every column...
		for (int j = 0; j < N; j++) {
			// For every element in the row-column pair
			int tmp = 0;
			for (int k = 0; k < N; k++) {
				// Accumulate the partial results
				tmp += h_A[i * N + k] * h_B[k * N + j];
			}

			// Check against the CPU result
			if(tmp != h_C[i * N + j]) {
				printf("Does not match!\n");
			}
		}
	}

	/*=============================Finish Test=================================*/

	//free(test_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
	cudaDeviceReset();
	return 0;
}