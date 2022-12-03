#include <stdio.h>
#include <stdlib.h>
#include <cassert>


#define N 2048
// tile size
#define TILE 8 // static allocation of shared memory 
__global__ void MatrixMultiply(float *a, float *b, float *c) {

    // calculate each threads global row and col
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    int start_row = row * TILE;
    int end_row = start_row + TILE;
    int start_col = col * TILE;
    int end_col = start_col + TILE;

    if(row < N && col < N) {
        for (int row = start_row; row < end_row; row++) {
            for(int col = start_col; col < end_col; col++) {
                float sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += a[row * N + k] * b[k * N + col];
                }
                c[row * N +col] = sum;
            }
        }
    } 
}

int main(){

    float h_A[N * N], h_B[N*N], h_C[N*N]; // host matrices

    // size to allocate
	size_t size = N * N * sizeof(float);

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
    
    // Set threads/block and blocks/thread
    int threads = 8; // 256 threads/block
	dim3 blockPerGrid(N / (TILE * 8), N / (TILE * 8)); // (X, Y, Z = 1)
	dim3 threadPerBlock(threads, threads); // thread.x = N, thread.y = N

    // timing
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    cudaEventRecord(start, 0);	
    // kernel launch
    MatrixMultiply<<<blockPerGrid, threadPerBlock>>>(d_A, d_B, d_C);
    //cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);	
	//cudaEventSynchronize(stop);
    
    //Read C from device
	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Elapsed time (ms): %f\n", milliseconds);

    //Calculate the MM result with normal CPU implementation and compare the results with the GPU

	float * test_C;
	test_C = (float *)malloc(size);	
	for (int i=0; i<N; i++){
		for (int j=0;j<N;j++){
			float sum = 0;
			for (int k=0;k<N;k++){
				float a = h_A[i*N+k];
				float b = h_B[k*N+j];
				sum += a*b;
			}
			test_C[i*N+j]= sum;
		}
	}
	int compare = 0;
	for(int i=0;i<N;i++){
		for(int j=0;j<N;j++){
			if (test_C[i*N+j]==h_C[i*N+j]){
				compare++;
			}else{
				compare+=0;
			}
		}
	}
	if(compare == N*N){
		printf("Success!\n");
	}else{
		printf("Error!\n");	
	}
    

	/*=============================Finish Test=================================*/

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
	cudaDeviceReset();
	return 0;
}