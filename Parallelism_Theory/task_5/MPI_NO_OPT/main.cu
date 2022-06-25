#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <math.h>


__global__ void iteration(double *a, double *b, int start, int end, int size_x, int size_y) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    
    // Does it belong to this GPU area
    if(idx*size_x + idy >= start && idx*size_x + idy < end)
        // Is it in array borders
        if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
            a[idx*size_x + idy] = (b[(idx+1)*size_x + idy] + b[(idx-1)*size_x + idy] + b[idx*size_x + (idy + 1)] + b[idx*size_x + (idy - 1)]) / 4;

    return;
}


__global__ void iterationMax(double *a, double *b, double* max, int size_x, int size_y) 
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < size_x - 1 && idy < size_y - 1 && idx >= 1 && idy >= 1) 
        max[idx*size_x + idy] = a[idx * size_x + idy] - b[idx * size_x + idy];


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



int main(int argc, char *argv[])
{
    if(argc != 4)
    {
        printf("1: GRID, 2: PRECISE, 3: ITER_COUNT\n");
        return 0;
    }

    const int GRID = atoi(argv[1]);
    const double PRECISE = atof(argv[2]);
    const int ITER_COUNT = atoi(argv[3]);
    
    int myRank, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    MPI_Status status;

    if(nRanks > 4) {
        printf("4 GPU is the local maximum!\n");
        return 1;
    }

    cudaSetDevice(myRank);

    int start, end;

    int step = GRID*GRID / nRanks;
    start = myRank * step;
    end = (myRank + 1) * step;

    double *dev_mass, *dev_mass_plus, *dev_max_mass, *dev_max;

    cudaMalloc((void**)&dev_mass, sizeof(double) * (GRID+2) * GRID);
    cudaMalloc((void**)&dev_mass_plus, sizeof(double) * (GRID+2) * GRID);
    cudaMalloc((void**)&dev_max_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_max, sizeof(double));

    auto free1 = dev_mass_plus;
    auto free2 = dev_mass;
    dev_mass_plus = dev_mass_plus + GRID;
    dev_mass = dev_mass + GRID;

    const dim3 BS = dim3(GRID/8, GRID/8);
    const dim3 GS = dim3(ceil(GRID/(float)BS.x), ceil(GRID/(float)BS.y));

    initMatrix<<< 1, 1 >>> (dev_mass_plus, dev_mass, GRID);

    double diff = 1;
    int k;


    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    
    for (k = 0; k < ITER_COUNT && diff > PRECISE; k++)
    {
        iteration<<< BS, GS >>> (dev_mass_plus, dev_mass, start, end, GRID, GRID);

        MPI_Sendrecv(dev_mass_plus + end - GRID, GRID, MPI_DOUBLE,
            (myRank+1)%nRanks,(myRank+1)%nRanks, dev_mass_plus + start - GRID, GRID,
            MPI_DOUBLE, (myRank-1 + nRanks)%nRanks, myRank, MPI_COMM_WORLD, &status);

        MPI_Sendrecv(dev_mass_plus + start, GRID, MPI_DOUBLE,
            (myRank-1 + nRanks)%nRanks,(myRank-1 + nRanks)%nRanks, dev_mass_plus + end, GRID,
            MPI_DOUBLE, (myRank+1)%nRanks, myRank, MPI_COMM_WORLD, &status);


        if(k % 100 == 99)
        {
            diff = 0;

            iterationMax<<< BS, GS >>> (dev_mass_plus, dev_mass, dev_max_mass, GRID, GRID);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID);
            cudaMemcpy(&diff, dev_max, sizeof(double), cudaMemcpyDeviceToHost);
            MPI_Allreduce(&diff, &diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        }

       
        
        auto tmp = dev_mass;
        dev_mass = dev_mass_plus;
        dev_mass_plus = tmp;

        if(myRank == 0)
            printf("Ops: %d, Err:%e\n", k, diff);
    }

    double *mass = (double*) calloc(GRID*GRID, sizeof(double));
    cudaMemcpy(mass, dev_mass, GRID*GRID*sizeof(double), cudaMemcpyDeviceToHost);

    // for(int i = 0; i < GRID; ++i) {
    //     for(int j = 0; j < GRID; ++j) {
    //         printf("%2.1f ", mass[i*GRID + j]);
    //     }
    //     printf("\n");
    // }

    cudaFree(free1);
    cudaFree(free2);
    cudaFree(dev_max_mass);
    cudaFree(dev_max);
    cudaFree(d_temp_storage);

    MPI_Finalize();

    return 0;
}
