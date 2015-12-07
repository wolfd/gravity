#include <stdio.h>

// There are ways to get this data but I'm too lazy
#define CUDA_CORES 384

#define N 1000 
//#define N 512 
#define THREADS_PER_BLOCK 512 

#define ITERATIONS 8000

#define GRAVITATIONAL_CONSTANT 66.7 // km^3 / (Yg * s^2)
//#define GRAVITATIONAL_CONSTANT 240300.0 // km^3 / (Yg * min^2)
#define TIME_STEP 3600.0 //
// http://www.wolframalpha.com/input/?i=gravitational+constant+in+km%5E3%2F%28Yg+*+s%5E2%29

void random_ints(int* a, int num) {
        int i;
        for(i = 0; i < num; ++i) {
                a[i] = rand();
        }
} 

void random_doubles(double* a, int num, double multiplier) {
        int i;
        for(i = 0; i < num; i++) {
                a[i] = (double)rand() / (double)RAND_MAX * multiplier;
        }
}

void random_double4s(double4* a, int num, double m0, double m1, double m2, double m3) {
        int i;
        for(i = 0; i < num; i++) {
                a[i].x = ((double)rand() / (double)RAND_MAX - 0.5) * m0;
                a[i].y = ((double)rand() / (double)RAND_MAX - 0.5) * m1;
                a[i].z = ((double)rand() / (double)RAND_MAX - 0.5) * m2;
                a[i].w = ((double)rand() / (double)RAND_MAX) * m3;
        }
}

void load_initial_data(double4 *in_pos, double4 *in_vel) {
        FILE *ifp;
        char *mode = "r";

        ifp = fopen("input.csv", mode);

        double x, y, z, xv, yv, zv;

        if(ifp == NULL) fprintf(stderr, "OH NO! No file!\n");

        int i = 0;
        while(fscanf(ifp, "%g, %g, %g, %g, %g, %g", &x, &y, &z, &xv, &yv, &zv)) {
                in_pos[i].x = x;
                in_pos[i].y = y;
                in_pos[i].z = z;

                in_vel[i].x = x;
                in_vel[i].y = y;
                in_vel[i].z = z;
                
                i++;
        }
        fclose(ifp);
}

__device__ double3 interaction(double4 body_a, double4 body_b, double3 accel) {
        double3 r;
        r.x = body_b.x - body_a.x;
        r.y = body_b.y - body_a.y;
        r.z = body_b.z - body_a.z;
 
        double dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + 4e6;
 
        //dist_sq += 4e6; // softening factor
 
        double inv_dist = rsqrt(dist_sq);
        double inv_dist_cube = inv_dist * inv_dist * inv_dist;
 
        double accel_total = GRAVITATIONAL_CONSTANT * body_b.w * inv_dist_cube;
 
        accel.x += r.x * accel_total;
        accel.y += r.y * accel_total;
        accel.z += r.z * accel_total;
        
        return accel;
}

__device__ double3 tile_calculation(double4 body_a, double3 accel) {
        int i;
        extern __shared__ double4 shared_positions[];
        //__shared__ double4 shared_positions[N];
        //double4 *shared_positions = SharedMemory();


#pragma unroll 128
        for(i = 0; i < blockDim.x; i++) {
                accel = interaction(body_a, shared_positions[i], accel);
        }

        return accel;
}

__device__ double4 calculate_accel(double4 *positions, int num_tiles) {
        extern __shared__ double4 shared_positions[];

        double4 cur_body; // current block's body

        int tile;

        double3 accel = {0.0, 0.0, 0.0};

        int gtid = blockIdx.x * blockDim.x + threadIdx.x;


        cur_body = positions[gtid];

        for(tile = 0; tile < num_tiles; tile++) {
                int idx = tile * blockDim.x + threadIdx.x;
                shared_positions[threadIdx.x] = positions[idx];
                __syncthreads();
#pragma unroll 128
                for(int counter = 0; counter < blockDim.x; counter++) {
                        accel = interaction(cur_body, shared_positions[counter], accel);
                }
                __syncthreads();
        }
        

        double4 accel4 = {accel.x, accel.y, accel.z, 0.0};
        return accel4;
}

__global__ void integrate(double4 *positions, double4 *vels, int num_tiles) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if(index >= N) {
                return;
        }

        double4 position = positions[index];

        //printf("what: %g, %g, %g, %g\n", position.x, position.y, position.z, position.w); 

        double4 accel = calculate_accel(positions, num_tiles);
        
        double4 velocity = vels[index]; 

        velocity.x += accel.x * TIME_STEP;
        velocity.y += accel.y * TIME_STEP;
        velocity.z += accel.z * TIME_STEP;

        position.x += velocity.x * TIME_STEP;
        position.y += velocity.y * TIME_STEP;
        position.z += velocity.z * TIME_STEP;

        __syncthreads();

        positions[index] = position;
        vels[index] = velocity;
}

int main(int argc, char *argv[]) {
                

        int block_size = N;

        int num_blocks = (N + block_size-1) / block_size;
        int num_tiles = (N + block_size - 1) / block_size;
        int shared_mem_size = block_size * 4 * sizeof(double); // 4 floats for pos

        double4 *positions, *vels;
        double4 *dev_positions, *dev_vels;

        int size = N * sizeof(double4);
        
        cudaMalloc((void**)&dev_positions, size);
        cudaMalloc((void**)&dev_vels, size);
         
        positions = (double4*)malloc(size);
        vels = (double4*)malloc(size);

//        load_initial_data(positions, vels);

        int seed = time(NULL);
        srand(seed);
        random_double4s(positions, N, 6e5, 6e5, 6e1, 11.6 * 2.0);
        random_double4s(vels, N, 0.5e2, 0.5, 0.1, 0.0);

        positions[0].w = 1.99e9;
        positions[0].y = 1.47e8;

        cudaMemcpy(dev_positions, positions, size, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_vels, vels, size, cudaMemcpyHostToDevice);


        FILE *fp = fopen("locations.csv", "w");

        for(int i = 0; i < ITERATIONS; i++) {
                integrate<<<num_blocks, block_size, shared_mem_size>>>(dev_positions, dev_vels, num_tiles);

                cudaMemcpy(positions, dev_positions, size, cudaMemcpyDeviceToHost);
                cudaMemcpy(vels, dev_vels, size, cudaMemcpyDeviceToHost);

                printf("%g\n", (double)i * 100.0 / (double)ITERATIONS);

                for(int j = 0; j < N; j++)
                fprintf(fp, "%d,%d,%g,%g,%g,%g,%g,%g\n", i, j, positions[j].x, positions[j].y, positions[j].z, vels[j].x, vels[j].y, vels[j].z);
        }

        fclose(fp);

        cudaFree(dev_positions);
        cudaFree(dev_vels);
        
        free(positions); free(vels);

        return 0;
}

