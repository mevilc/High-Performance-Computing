#include <stdio.h>
#include <stdlib.h>
#include <cassert>


#define N 2048

// tile size. CHANGE THIS TO 8 x 8.
#define SHMEM_SIZE (16 * 16) // static allocation of shared memory 

__global__ void MatrixMultiply(float *a, float *b, float *c) {
    // Allocate shared memory per thread block
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // calculate each threads global row and col
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Extract builin values to simplify code
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int dim = blockDim.x;

    // Move tile across length of the grid
    if (row < N && col < N) {
        for (int i = 0; i < (N / dim); i++) {
            A[ty * dim + tx] = a[(row * N) + (i * dim) + tx];
            B[ty * dim + tx] = b[(i * dim * N) + (ty * N) + col];
            __syncthreads(); // make sure all shared memory is loaded

            // splitting NxN mat mul into dimxdim mat muls
            // Acumulate partial results

            for (int j = 0; j < dim; j++) {
                //tmp += A[ty * dim + j] * B[j * dim + tx];
                c[row * N + col] += A[ty * dim + j] * B[j * dim + tx];
            }
            __syncthreads(); // finish accumulating results before re populating
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
    int blocks = N / threads;
	dim3 blockPerGrid(blocks, blocks); // (X, Y, Z = 1)
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