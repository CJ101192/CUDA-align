#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>
#include <assert.h>

#include <SW_kernel_1.cu>
#include <time.h>

void Pairwise_Alignment (int *h_A, int *d_B, int *h_B, int *h_Max_CPU_All, int *h_A_Location_All, int *h_B_Location_All,
						 int *h_Max_CPU, int *h_A_Location, int *h_B_Location,
						 int K1R, int MyProc, int NumProcs, int L_B, int L_A, int L_A1,  
						 int si, int dis, int Gop, int Gex, int L_M, int Threads_N, int Start_A,
						 int DATA_SZ_K1, int DATA_SZ_A, int DATA_SZ_H, int DATA_SZ_B, int DATA_SZ_M, int DATA_SZ_L)
{
    /*  Given values :
            K1R (Top K1R scores need to be reported) ; Current value:256
            L_B and L_A are lengths of query and DB sequence
            DATA_SZ_K1 = K1R * sizeof(int)
            DATA_SZ_A  = L_A * sizeof(int)
            DATA_SZ_B  = L_B * sizeof(int)
            DATA_SZ_H  = 2*(L_A+1) * sizeof(int)
            si, dis, Gop, Gex are match, mismatch, gap initiation, gap extension scores respectively
            d_B points to device's copy of query sequence (initialised in Main.cu : ~125)
            h_B points to host's copy of query sequence 
            h_A points to host's copy of database sequence
            MyProc is the MPI rank of this CPU
            Threads_N (initialised at Main.cu:114 for use in SW_kernel_1_old) : Grids(64)*Blocks(1024) <ignore>
            NumProcs  : Total CPU processes spawned on MPI
            L_M = min(Threads_N,L_A) +1  (May be, no. of horizontal rows we can deal with)
            DATA_SZ_M : L_M * sizeof(int)
            DATA_SZ_L : Space to store scores computed by seperate threads
    */

    /*  To return 
            h_A_Location_All and h_B_Location_All {allocated size: DATA_SZ_K1 each} should contain cell positions of reported sequences
            h_Max_CPU {allocated size : DATA_SZ_K1} should contain maximum scores on this CPU
            h_Max_CPU_All {allocated size : DATA_SZ_K1} should contain maximum scores of all CPUs

    */
    int *d_A , maxScore = 0;
   
    /*For debugging */
    cudaPrintfInit();
    
    /*
    //For verification
    if (MyProc == 0)
    {
        printf("...Kernel_1 on CPU begins.\n");
        int Max_CPU = 0;
        Kernel_1_CPU(h_A, h_B, Max_CPU, L_A1, L_B, si, dis, Gop, Gex);
        int Max_CPU_Reduced;
        MPI_Reduce(&Max_CPU, &Max_CPU_Reduced, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("Maximum score on CPU            :   %d \n", Max_CPU_Reduced);
    }
    */

    if (MyProc==0)
    {
        printf("...Allocating GPU memory.\n");
    }

    //Global memory to hold main sequence
    CUDA_SAFE_CALL(cudaMalloc((void**)&d_A, DATA_SZ_A));
    CUDA_SAFE_CALL(cudaMemcpy(d_A, h_A, DATA_SZ_A, cudaMemcpyHostToDevice));

    //Global memory to hold query sequence
    CUDA_SAFE_CALL(cudaMemcpy(d_B, h_B, DATA_SZ_B, cudaMemcpyHostToDevice));

    //Decide number of blocks and partition the work (assuming overlap size of L_B)
    int NO_THREADS = (L_A < 3000000) ? 16384 : 131072;
    NO_THREADS = (NO_THREADS/L_B)*L_B;

    assert(L_B >= 32);
    assert(L_B <= 1024);
    assert(NO_THREADS % L_B == 0);
    int noBlocks = NO_THREADS/L_B;
    int Length = (L_A- L_B)/noBlocks;
    int L_A_ = Length + L_B;                  //Lenght of main sequence each Block works on

    //Count time taken by Kernel 1

    cudaEvent_t Kernel1OverallTimeStart, Kernel1OverallTimeStop;
    cudaEventCreate(&Kernel1OverallTimeStart); 
    cudaEventCreate(&Kernel1OverallTimeStop);
    cudaEventRecord(Kernel1OverallTimeStart,0);

    //Global memory to hold intermediate scores (1 buffer for vertical scores and 2 for main scores)
    int *maxScoresPerBlock;
    CUDA_SAFE_CALL(cudaMalloc((void**)&maxScoresPerBlock, noBlocks*sizeof(int)));

    dim3 BlockSize(L_B,1);
    dim3 GridSize(noBlocks,1);

    if (MyProc==0)
    {
        printf("...Kernel_1 on GPU begins with %d blocks and %d threads each.\n", noBlocks, L_B);
        printf("Width of each block             :   %d \n", L_A_); 
    }
    //Call the kernel for computing maxScore
    diagonalComputation<<<GridSize, BlockSize , DATA_SZ_B*4>>>(L_A_, L_B, d_A, d_B,
                                                maxScoresPerBlock, 
                                                Length, 
                                                si, dis, Gop, Gex); 

    
    //Copying scores from GPU to CPU
    int *maxScoresPerBlockCPU = (int *)malloc(noBlocks * sizeof(int));
	CUDA_SAFE_CALL( cudaMemcpy( maxScoresPerBlockCPU, maxScoresPerBlock  , noBlocks*sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i=0 ; i<noBlocks; i++)
    {
        maxScore = max(maxScore, maxScoresPerBlockCPU[i]);
    }

    //Aggregate maximum scores from all CPUs
    int maxScoreReduced;
    MPI_Reduce(&maxScore, &maxScoreReduced, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    

    if (MyProc==0)
    {
        cudaEventRecord(Kernel1OverallTimeStop, 0);
        cudaEventSynchronize(Kernel1OverallTimeStop);
        float Kernel1TotalElapsedTime;
        cudaEventElapsedTime(&Kernel1TotalElapsedTime, Kernel1OverallTimeStart, Kernel1OverallTimeStop);

		printf("Total time spent in Kernel 1    :   %f ms \n", Kernel1TotalElapsedTime);
        printf("Maximum score by GPUs           :   %d \n", maxScoreReduced);
    }
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
}
