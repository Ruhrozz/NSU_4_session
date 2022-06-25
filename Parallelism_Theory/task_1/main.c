#include <stdio.h>
#include <math.h>

#define PGI_ACC_TIME 1

double mass[1000000000];

int main() {
    double s = 0;
#pragma acc data copyin(mass)
#pragma acc kernels
    for(int i = 0; i < 10000000; i++) {
        mass[i] = sin(i*M_PI/2);
        s += mass[i];
    }
#pragma acc data copyout(s) 
    printf("%e\n", s);
    return 0;
}



