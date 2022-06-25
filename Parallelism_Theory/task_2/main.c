#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

    double fin_sum = 1;
    double sum = 1;
    int k;

    #pragma acc data copy(mass[0:GRID*GRID], mass_plus[0:GRID*GRID], sum)
    {
        for (k = 0; k < ITER_COUNT && sum > PRECISE; k++)
        {
            sum = 1;

            if(k % 1 == 0)
            {
                #pragma acc kernels
                sum = 0;
            }

            #pragma acc data present(mass[0:GRID*GRID], mass_plus[0:GRID*GRID])
            {
                if(k % 1 == 0)
                {
                    #pragma acc parallel loop independent collapse(2) reduction(max:sum)
                    for (int i = 1; i < GRID - 1; i++)
                    {
                        for (int j = 1; j < GRID - 1; j++)
                        {
                            mass_plus[i * GRID + j] = (mass[(i + 1) * GRID + j] + mass[(i - 1) * GRID + j] + mass[i * GRID + j + 1] + mass[i * GRID + j - 1]) / 4;
                            fin_sum = sum = fmax(sum, mass_plus[i * GRID + j] - mass[i * GRID + j]);
                        }
                    }
                }
                else
                {
                    #pragma acc parallel loop independent collapse(2)
                    for (int i = 1; i < GRID - 1; i++)
                    {
                        for (int j = 1; j < GRID - 1; j++)
                        {
                            mass_plus[i * GRID + j] = (mass[(i + 1) * GRID + j] + mass[(i - 1) * GRID + j] + mass[i * GRID + j + 1] + mass[i * GRID + j - 1]) / 4;
                        }
                    }
                }
            }   

            tmp = mass;
            mass = mass_plus;
            mass_plus = tmp;
            #pragma acc update self(sum) if(k % 1 == 0)
        }
    printf("err: %e, iter: %d\n", fin_sum, k);
    }
    return 0;
}
