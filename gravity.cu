#include <stdio.h>

#define N 2048 
#define THREADS_PER_BLOCK N 

#define GRAVITATIONAL_CONSTANT 66.7 // km^3 / (Yg * s^2)
// http://www.wolframalpha.com/input/?i=gravitational+constant+in+km%5E3%2F%28Yg+*+s%5E2%29

__constant__ double G;

void random_ints(int* a, int num) {
        int i;
        for(i = 0; i < num; ++i) {
                a[i] = rand();
        //        a[i] = 1;
        }
} 

void random_doubles(double* a, int num) {
        int i;
        for(i = 0; i < num; ++i) {
                a[i] = (double)rand() / (double)RAND_MAX;
        }
} 

__global__ void add(int *a, int *b, int *c) {
        int index = threadIdx.x + blockIdx.x * blockDim.x;
        c[index] = a[index] + b[index];
        //*c = *a + *b;
}

__global__ void dot(int *a, int *b, int *c) {
        __shared__ int temp[THREADS_PER_BLOCK];

        int index = threadIdx.x + blockIdx.x * blockDim.x;

        temp[threadIdx.x] = a[index] * b[index];

        __syncthreads();

        if(0 == threadIdx.x) {
                int sum = 0;
                for(int i = 0; i < THREADS_PER_BLOCK; i++) {
                        sum += temp[i];
                }
                atomicAdd(c, sum);
        }
}

__global__ void gravity(double *x, double *y, double *m, double *accel) {
        __shared__ double accel_parts[THREADS_PER_BLOCK];
        int index = threadIdx.x + blockIdx.x * blockDim.x;

        double d_x = x[blockIdx.x] - x[threadIdx.x];
        double d_y = y[blockIdx.x] - y[threadIdx.x];

        double dist_sq = d_x * d_x + d_y * d_y; 

        accel_parts[threadIdx.x] = 66.7 * m[threadIdx.x] / dist_sq; 

        /*
           G * M * m
             -----
            dist^2
        */

        __syncthreads();

        if(0 == threadIdx.x) {
                double sum = 0.0;

                for(int i = 0; i < blockDim.x; i++) {
                        sum += accel_parts[i];
                }
                accel[blockIdx.x] = sum;
        }
}


int main(void) {
        double *x, *y, *m, *acc;
        double *dev_x, *dev_y, *dev_m, *dev_acc;
        int size = N * sizeof(double);

        /*cudaMemcpyToSymbol(G,
                        GRAVITATIONAL_CONSTANT,
                        sizeof(double),
                        0,
                        cudaMemcpyHostToDevice);
        */

        cudaMalloc((void**)&dev_x, size);
        cudaMalloc((void**)&dev_y, size);
        cudaMalloc((void**)&dev_m, size);
        cudaMalloc((void**)&dev_acc, size);

        x = (double*)malloc(size);
        y = (double*)malloc(size);
        m = (double*)malloc(size);
        acc = (double*)malloc(size);

        random_doubles(x, N);
        random_doubles(y, N);
        random_doubles(m, N);

        cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y, y, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_m, m, size, cudaMemcpyHostToDevice);

        gravity<<<N, THREADS_PER_BLOCK>>>(dev_x, dev_y, dev_m, dev_acc);

        cudaMemcpy(acc, dev_acc, size, cudaMemcpyDeviceToHost);

        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_m);
        cudaFree(dev_acc);
        
        free(x); free(y); free(m); 

        printf("Numbers:\n");
        for(int i = 0; i < N; i++) {
                printf("%d\n", acc[i]);
        }

        free(acc);

        return 0;
}

