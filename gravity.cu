#include <stdio.h>

#define N 512 
#define THREADS_PER_BLOCK 512 

//#define GRAVITATIONAL_CONSTANT 66.7 // km^3 / (Yg * s^2)
#define GRAVITATIONAL_CONSTANT 240300.0 // km^3 / (Yg * min^2)
// http://www.wolframalpha.com/input/?i=gravitational+constant+in+km%5E3%2F%28Yg+*+s%5E2%29

__constant__ double G;

void random_ints(int* a, int num) {
        int i;
        for(i = 0; i < num; ++i) {
                a[i] = rand();
        //        a[i] = 1;
        }
} 

void random_doubles(double* a, int num, double multiplier) {
        int i;
        for(i = 0; i < num; i++) {
                a[i] = (double)rand() / (double)RAND_MAX * multiplier;
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

__global__ void update_positions(double *x, double *y, double *vx, double *vy) {
        x[blockIdx.x] += vx[blockIdx.x] * 1.0;
        y[blockIdx.x] += vy[blockIdx.x] * 1.0;
}

__global__ void gravity(double *x, double *y, double *m, double *vx, double *vy) {
        __shared__ double accel_parts[THREADS_PER_BLOCK];
        //int index = threadIdx.x + blockIdx.x * blockDim.x;

        double d_x, d_y, dist_sq;

        if(blockIdx.x != threadIdx.x) {
                d_x = x[blockIdx.x] - x[threadIdx.x];
                d_y = y[blockIdx.x] - y[threadIdx.x];

                dist_sq = d_x * d_x + d_y * d_y; 
                
                if(dist_sq > 10) {
                        accel_parts[threadIdx.x] = 66.7 * m[threadIdx.x] / dist_sq;
                } else {
                        accel_parts[threadIdx.x] = 0.0;
                }
        } else {
                accel_parts[threadIdx.x] = 0.0;
        }

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
                vx[blockIdx.x] = sum * (d_x / sqrt(dist_sq));
                vy[blockIdx.x] = sum * (d_y / sqrt(dist_sq));
        }
}

int main(void) {
        double *x, *y, *m, *vx, *vy;
        double *dev_x, *dev_y, *dev_m, *dev_vx, *dev_vy;
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
        cudaMalloc((void**)&dev_vx, size);
        cudaMalloc((void**)&dev_vy, size);

        x = (double*)malloc(size);
        y = (double*)malloc(size);
        m = (double*)malloc(size);
        vx = (double*)malloc(size);
        vy = (double*)malloc(size);

        int seed = time(NULL);
        srand(seed);

        random_doubles(x, N, 5e4);
        random_doubles(y, N, 5e4);
        random_doubles(m, N, 0.5);

        cudaMemcpy(dev_x, x, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_y, y, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_m, m, size, cudaMemcpyHostToDevice);

        FILE *fp = fopen("locations.csv", "w");


        for(int i = 0; i < 100; i++) {
                update_positions<<<N, 1>>>(dev_x, dev_y, dev_vx, dev_vy);
                gravity<<<N, THREADS_PER_BLOCK>>>(dev_x, dev_y, dev_m, dev_vx, dev_vy);

                cudaMemcpy(x, dev_x, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(y, dev_y, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(vx, dev_vx, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(vy, dev_vy, size, cudaMemcpyDeviceToHost);

                //printf("%g\n", (double)i / (double)ITERATIONS);

                for(int j = 0; j < N; j++)
                fprintf(fp, "%d, %d, %g, %g, %g, %g\n", i, j, x[j], y[j], vx[j], vy[j]);
        }

        fclose(fp);

        cudaFree(dev_x);
        cudaFree(dev_y);
        cudaFree(dev_m);
        cudaFree(dev_vx);
        cudaFree(dev_vy);
        
        free(x); free(y); free(m);

        //printf("Numbers:\n");
        //for(int i = 0; i < N; i++) {
        //        printf("%d: (%g, %g)\n", i, vx[i], vy[i]);
        //}
        
        free(vx); free(vy);

        return 0;
}

