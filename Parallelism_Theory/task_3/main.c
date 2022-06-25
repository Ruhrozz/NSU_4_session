#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENACC
#include "/opt/nvidia/hpc_sdk/Linux_x86_64/21.11/math_libs/11.5/targets/x86_64-linux/include/cublas_v2.h"
#endif


int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        printf("1: GRID, 2: PRECISE, 3: ITER_COUNT\n");
        return 0;
    }

    const int GRID = atoi(argv[1]);
    const double PRECISE = atof(argv[2]);
    const int ITER_COUNT = atoi(argv[3]);

    double *mass;
    double *mass_plus;
    double *tmp;

    mass = (double*) calloc(GRID*GRID, sizeof(double));
    mass_plus = (double*) calloc(GRID*GRID, sizeof(double));

    mass[0] = 10;
    mass[(GRID-1)*GRID +GRID-1] = 30;
    mass[GRID-1] = 20;
    mass[(GRID-1)*GRID] = 20;

    mass_plus[0] = 10;
    mass_plus[(GRID-1)*GRID +GRID-1] = 30;
    mass_plus[GRID-1] = 20;
    mass_plus[(GRID-1)*GRID] = 20;

    for(int i = 1; i < GRID - 1; ++i)
    {
        mass[i] = 10.0 + 10.0 * i/(GRID - 1.0);
        mass[i*GRID] = 10.0 + 10.0 * i / (GRID - 1.0);
        mass[i*GRID + GRID-1] = 20.0 + 10.0 * i / (GRID - 1.0);
        mass[(GRID-1)*GRID + i] = 20.0 + 10.0 * i / (GRID - 1.0);
        mass_plus[i] = 10.0 + 10.0 * i/(GRID - 1.0);
        mass_plus[i*GRID] = 10.0 + 10.0 * i / (GRID - 1.0);
        mass_plus[i*GRID + GRID-1] = 20.0 + 10.0 * i / (GRID - 1.0);
        mass_plus[(GRID-1)*GRID + i] = 20.0 + 10.0 * i / (GRID - 1.0);
    }
#ifdef _OPENACC
    cublasStatus_t status;
    cublasHandle_t handle;
    int index;
    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
	    printf("exit_failure\n");
        return EXIT_FAILURE;
    }
#endif

    double sum = 1;
    int k;
    double fin_err = -1;

    #pragma acc data copy(mass[0:GRID*GRID], mass_plus[0:GRID*GRID], sum)
    {
        for (k = 0; k < ITER_COUNT && sum > PRECISE; k++)
        {
            sum = 1;
            #pragma acc data present(mass[0:GRID*GRID], mass_plus[0:GRID*GRID])
            #pragma acc parallel loop independent collapse(2)
            for (int i = 1; i < GRID - 1; i++)
            {
                for (int j = 1; j < GRID - 1; j++)
                {
                    mass_plus[i * GRID + j] = (mass[(i + 1) * GRID + j] + mass[(i - 1) * GRID + j] + mass[i * GRID + j + 1] + mass[i * GRID + j - 1]) / 4;
                }
            }

            if(k % 100 == 0)
            {
                #ifdef _OPENACC
                    #pragma acc host_data use_device(mass, mass_plus)
                    {
                        const double minus_one = -1;
                        status = cublasDaxpy(handle, GRID*GRID, &minus_one, mass, 1, mass_plus, 1);
                        status = cublasIdamax(handle, GRID*GRID, mass_plus, 1, &index);
                    }

                    #pragma acc update self(mass_plus[0:GRID*GRID])

                    fin_err = sum = mass_plus[index-1];

                    #pragma acc host_data use_device(mass, mass_plus)
                    {
                        const double one = 1;
                        status = cublasDaxpy(handle, GRID*GRID, &one, mass, 1, mass_plus, 1);
                    }

                    #pragma acc update self(mass_plus[0:GRID*GRID])
                #else
                    sum = 0;
                    for(int i = 1; i < GRID-1; ++i)
                    {
                        for(int j = 1; j < GRID-1; ++j)
                        {
                            fin_err = sum = fmax(sum, mass_plus[i*GRID + j] - mass[i*GRID + j]);
                        }
                    }
                #endif
            }

            tmp = mass;
            mass = mass_plus;
            mass_plus = tmp;
        }

        printf("\nerr: %e, iter: %d\n", fin_err, k);
    }

    return 0;
}
