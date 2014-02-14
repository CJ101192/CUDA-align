default:
	nvcc -g -G Main.cu -o SW_exe -I./../include/oldCUDAlibs/common/inc/ -I./ -I//usr/mpi/gcc/openmpi-1.4.3/include/ -L./../include/oldCUDAlibs/common/lib/ -L/home/tarun/Installations/openmpi-1.4.5-multithreaded/release/lib -Xcompiler -lmpi -lpthread -lrt -lcutil_x86_64 $(ARGS)

#Makefile to run on gpuCluster
#Included nvcc path /usr/local/cuda/bin/ in bash_profile
#For execution:
#Run mpirun by path /usr/mpi/gcc/openmpi-1.4.3/bin/mpirun 
#include /usr/local/cuda/lib64/ in LD_LIBRARY_PATH
#while running on multiple nodes, add -x LD_LIBRARY_PATH

