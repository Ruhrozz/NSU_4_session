compile:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lnccl -lmpi -arch sm_70 main.cu -o test

run: compile
	UCX_WARN_UNUSED_ENV_VARS=n mpirun -mca pml ucx -x UCX_TLS=cuda,sm,posix -np 3 ./test 16 0.000001 1000000
	