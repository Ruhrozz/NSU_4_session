#include <mpi.h>
#include <string.h>
#include <stdio.h>
#include <cuda.h>
#include <cub/cub.cuh>
#include <math.h>
#include <nccl.h>


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

    if(nRanks > 4) {
        printf("GPU max count == 4!\n");
        return 1;
    }


    ncclUniqueId id;
    if (myRank == 0) 
        ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);


    cudaSetDevice(myRank);
    
    int start, end;

    int step = GRID*GRID / nRanks;
    start = myRank * step;
    end = (myRank + 1) * step;


    double *dev_mass, *dev_mass_plus, *dev_max_mass, *dev_max;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMalloc((void**)&dev_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_mass_plus, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_max_mass, sizeof(double) * GRID * GRID);
    cudaMalloc((void**)&dev_max, sizeof(double));

    const dim3 BS = dim3(GRID/8, GRID/8);
    const dim3 GS = dim3(ceil(GRID/(float)BS.x), ceil(GRID/(float)BS.y));

    initMatrix<<< 1, 1 , 0, stream>>> (dev_mass_plus, dev_mass, GRID);

    double diff = 1;
    int k;


    ncclComm_t comm;
    ncclCommInitRank(&comm, nRanks, id, myRank);


    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID, stream);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    
    for (k = 0; k < ITER_COUNT && diff > PRECISE; k++)
    {
        iteration<<< BS, GS, 0, stream >>> (dev_mass_plus, dev_mass, start, end, GRID, GRID);

        ncclGroupStart();
        // Bottom
        if(myRank != nRanks-1)
            ncclSend(dev_mass_plus + end - GRID, GRID, 
                        ncclDouble, (myRank+1)%nRanks, 
                        comm, stream);
        if(myRank != 0)
            ncclRecv(dev_mass_plus + start - GRID, GRID, 
                        ncclDouble, (myRank-1 + nRanks)%nRanks, 
                        comm, stream);

        // ncclGroupStart();
        // ncclGroupEnd();
        // Top
        if(myRank != 0)
            ncclSend(dev_mass_plus + start, GRID, 
                        ncclDouble, (myRank-1)%nRanks, 
                        comm, stream);
        if(myRank != nRanks-1)
            ncclRecv(dev_mass_plus + end, GRID, 
                        ncclDouble, (myRank+1 + nRanks)%nRanks, 
                        comm, stream);

        ncclGroupEnd();
        
        if(k % 100 == 99)
        {
            diff = 0;

            iterationMax<<< BS, GS, 0, stream >>> (dev_mass_plus, dev_mass, dev_max_mass, GRID, GRID);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, dev_max_mass, dev_max, GRID*GRID, stream);
            ncclAllReduce(dev_max, dev_max, 1, ncclDouble, ncclMax, comm, stream);
            
            cudaMemcpyAsync(&diff, dev_max, sizeof(double), cudaMemcpyDeviceToHost, stream);

            if(myRank == 0)
                printf("Ops: %d, Err:%e\n", k, diff);
        }

        
        auto tmp = dev_mass;
        dev_mass = dev_mass_plus;
        dev_mass_plus = tmp;

    }

    double *mass = (double*) calloc(GRID*GRID, sizeof(double));
    cudaMemcpyAsync(mass, dev_mass, GRID*GRID*sizeof(double), cudaMemcpyDeviceToHost, stream);

    

    // for(int i = 0; i < GRID; ++i) {
    //     for(int j = 0; j < GRID; ++j) {
    //         printf("%2.1f ", mass[i*GRID + j]);
    //     }
    //     printf("\n");
    // }

    ncclCommDestroy(comm);
    cudaFree(dev_mass);
    cudaFree(dev_mass_plus);
    cudaFree(dev_max_mass);
    cudaFree(dev_max);
    cudaStreamDestroy(stream);

    MPI_Finalize();

    return 0;
}
