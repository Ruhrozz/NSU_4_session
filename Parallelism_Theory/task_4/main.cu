#include <cuda.h>
// #include <iostream>
#include <cub/cub.cuh>
#include <math.h>


__global__ void iteration(double *a, double *b, int size_x, int size_y) 
{
        __shared__ double tmp[32][32];
        __shared__ double _tmp[32][32];

        int idx = blockIdx.x*blockDim.x + threadIdx.x;
        int idy = blockIdx.y*blockDim.y + threadIdx.y;

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
            tmp[threadIdx.x][threadIdx.y] = 0;

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
            tmp[threadIdx.x][threadIdx.y] += b[(idx+1)*size_x + idy];
            tmp[threadIdx.x][threadIdx.y] += b[(idx-1)*size_x + idy];
            tmp[threadIdx.x][threadIdx.y] += b[idx*size_x + (idy + 1)];
            tmp[threadIdx.x][threadIdx.y] += b[idx*size_x + (idy - 1)];

        __syncthreads();

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
        {
            a[idx*size_x + idy]  = tmp[threadIdx.x][threadIdx.y] / 4;
        }

        __syncthreads();

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
            _tmp[threadIdx.x][threadIdx.y] = 0;

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
            _tmp[threadIdx.x][threadIdx.y] += a[(idx+1)*size_x + idy];
            _tmp[threadIdx.x][threadIdx.y] += a[(idx-1)*size_x + idy];
            _tmp[threadIdx.x][threadIdx.y] += a[idx*size_x + (idy + 1)];
            _tmp[threadIdx.x][threadIdx.y] += a[idx*size_x + (idy - 1)];

        __syncthreads();

        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
        {
            b[idx*size_x + idy]  = _tmp[threadIdx.x][threadIdx.y] / 4;
        }


    return;
}


__global__ void iterationMax(double *a, double *b, double* max, int size_x, int size_y) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1)
    {
        a[idx*size_x + idy] = (b[(idx+1)*size_x + idy] + b[(idx-1)*size_x + idy] + b[idx*size_x + (idy + 1)] + b[idx*size_x + (idy - 1)]) / 4;
        max[idx*size_x + idy] = a[idx * size_x + idy] - b[idx * size_x + idy];
    }

    return;
}

__global__ void initMatrix(double *mass1, double *mass2, int size) 
{
    mass1[0] = 10;
    mass1[(size-1)*size +size-1] = 30;
    mass1[size-1] = 20;
    mass1[(size-1)*size] = 20;

    mass2[0] = 10;
    mass2[(size-1)*size +size-1] = 30;
    mass2[size-1] = 20;
    mass2[(size-1)*size] = 20;

    for(int i = 1; i < size - 1; ++i)
    {
        mass1[i] = 10.0 + 10.0 * i/(size - 1.0);
        mass1[i*size] = 10.0 + 10.0 * i / (size - 1.0);
        mass1[i*size + size-1] = 20.0 + 10.0 * i / (size - 1.0);
        mass1[(size-1)*size + i] = 20.0 + 10.0 * i / (size - 1.0);
        mass2[i] = 10.0 + 10.0 * i/(size - 1.0);
        mass2[i*size] = 10.0 + 10.0 * i / (size - 1.0);
        mass2[i*size + size-1] = 20.0 + 10.0 * i / (size - 1.0);
        mass2[(size-1)*size + i] = 20.0 + 10.0 * i / (size - 1.0);
    }
}


int main(int argc, char* argv[])
{
    // if(argc != 4)
    // {
    //     printf("1: GRID, 2: PRECISE, 3: ITER_COUNT\n");
    //     return 0;
    // }

    // const int GRID = atoi(argv[1]);
    // const double PRECISE = atof(argv[2]);
    // const int ITER_COUNT = atoi(argv[3]);

    const int GRID = 128;
    const double PRECISE = 0.000001;
    const int ITER_COUNT = 1000000;

    double *dev_mass, *dev_mass_plus, *dev_max_mass, *dev_max;

    cudaMalloc((void**)&dev_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_mass_plus, sizeof(double) * GRID * GRID);

    const dim3 BS = dim3(GRID/8, GRID/8);
    const dim3 GS = dim3(ceil(GRID/(float)BS.x), ceil(GRID/(float)BS.y));

    initMatrix<<< 1, 1 >>> (dev_mass_plus, dev_mass, GRID);

    cudaMalloc((void**)&dev_max_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_max, sizeof(double));

    int k;
    double diff = 1;

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    for (k = 0; k < ITER_COUNT && diff > PRECISE; k++)
    {
        if(k % 100 == 99)
        {
            diff = 0;

            iterationMax<<< BS, GS >>> (dev_mass_plus, dev_mass, dev_max_mass, GRID, GRID);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID);
            cudaMemcpy(&diff, dev_max, sizeof(double), cudaMemcpyDeviceToHost);
        }
        else
        {          
            iteration<<< BS, GS >>> (dev_mass_plus, dev_mass, GRID, GRID);
        }
    }

    printf("err: %e, iter: %d\n", diff, k*2);

    cudaFree(dev_mass);
    cudaFree(dev_mass_plus);
    cudaFree(dev_max);
    cudaFree(dev_max_mass);

    return 0;
}
