#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>


#include <Scan_SW.cu>  
#include <ss.cu>
#include <SW_kernel_1.cu>
#include <time.h>

void Pairwise_Alignment (int *h_A, int *d_B, int *h_B, int *h_Max_CPU_All, int *h_A_Location_All, int *h_B_Location_All,
						 int *h_Max_CPU, int *h_A_Location, int *h_B_Location,
						 int K1R, int MyProc, int NumProcs, int L_B, int L_A, int L_A1,  
						 int si, int dis, int Gop, int Gex, int L_M, int Threads_N, int Start_A,
						 int DATA_SZ_K1, int DATA_SZ_A, int DATA_SZ_H, int DATA_SZ_B, int DATA_SZ_M, int DATA_SZ_L)
{
	int	*h_H, *h_Max_GPU, *h_Loc_GPU, *h_Con_New, *h_Loc_GPU2, *h_Max_GPU2,  *Max_CPU;
	int *d_H, *d_F, *d_E_til, *d_H_til, *d_Max_H, *d_Loc_H, *d_Con_Old,  *d_Con_New, *d_Max_H2, *d_Loc_H2;
	int *d_A;
	float Total_Copy = 0;
    unsigned int hTimer;
	dim3 BlockSize(64, 1);         //64
    dim3 GridSize (512, 1);        // 512

	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );


	if (MyProc == 0)
	{
		printf("...Allocating CPU memory.\n");
	}
    //Profiling Data initialization time
    cudaEvent_t dataInitStart, dataInitStop, Kernel1OverallTimeStart, Kernel1OverallTimeStop;

    cudaEventCreate(&dataInitStart); 
    cudaEventCreate(&dataInitStop);
    cudaEventCreate(&Kernel1OverallTimeStart); 
    cudaEventCreate(&Kernel1OverallTimeStop);

    cudaEventRecord(dataInitStart,0);
    cudaEventRecord(Kernel1OverallTimeStart,0);

	h_H               = (int *)malloc(DATA_SZ_H);
	Max_CPU           = (int *)malloc(DATA_SZ_K1);
	h_Max_GPU         = (int *)malloc(DATA_SZ_A);
	h_Max_GPU2        = (int *)malloc(DATA_SZ_A);
	h_Loc_GPU2        = (int *)malloc(DATA_SZ_A);
	h_Loc_GPU         = (int *)malloc(DATA_SZ_A);
	h_Con_New         = (int *)malloc(sizeof(int));

	if (MyProc == 0)
	{
		printf("...Allocating GPU memory.\n");
	}
	CUDA_SAFE_CALL( cudaMalloc((void **)&d_A,       DATA_SZ_A) );                //////////////
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_H,       DATA_SZ_H) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_F,       DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_E_til,   DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_H_til,   DATA_SZ_A) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Con_Old, DATA_SZ_M) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Con_New, DATA_SZ_M) );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_H,   DATA_SZ_L) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_H,   DATA_SZ_L) );

    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Max_H2,  DATA_SZ_A/4) );
    CUDA_SAFE_CALL( cudaMalloc((void **)&d_Loc_H2,  DATA_SZ_A/4) );
    
    for(int i = 0; i < 2*(L_A+1) ; i++)
    {
		h_H[i]=0;
    }  
    for(int i = 0; i < (L_A) ; i++)
    {
		h_Max_GPU[i]=0;
		h_Loc_GPU[i]=0;
    }  
 
    for(int i = 0; i < K1R ; i++)
    {
		h_A_Location[i]=0;
		h_B_Location[i]=0;
		Max_CPU[i]=1;
		h_Max_CPU[i]=1;
	}

	if (MyProc == 0)
	{
		printf("...Copying input data to GPU memory \n");
	}
    
	CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, DATA_SZ_A, cudaMemcpyHostToDevice) );                   //////////////////
    CUDA_SAFE_CALL( cudaMemcpy(d_F, h_H, DATA_SZ_A, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_H, h_H, DATA_SZ_H, cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, DATA_SZ_B, cudaMemcpyHostToDevice) );

	if (MyProc == 0)
	{
		printf("Data initialization done.\n");
		printf("-------------------------------------------------------------\n");
	}
    cudaEventRecord(dataInitStop, 0);
    cudaEventSynchronize(dataInitStop);
    float dataInitElapsedTime;
    cudaEventElapsedTime(&dataInitElapsedTime, dataInitStart, dataInitStop);


    if (MyProc==0)
    {
		printf("Memory Allocation done in       :   %f ms \n", dataInitElapsedTime);
		printf("-------------------------------------------------------------\n");
    }




	
	
	//************************************* SW in CPU ***********************************
	/*float Start_CPU_SW_Time=0;
	float End_CPU_SW_Time=0;
	if ((MyProc == 0) )
	{
//	printf("------------------------------------- \n");
//	printf("CPU execution started...\n");

	Start_CPU_SW_Time =	cutGetTimerValue(hTimer);
	//Kernel_1_CPU(h_A,h_B,Max_CPU,A_Location_CPU, B_Location_CPU, K1R,L_A1,L_B,si, dis, Gop, Gex);  					  
	End_CPU_SW_Time =	cutGetTimerValue(hTimer);
	
//	printf("CPU execution finished...\n");
//   printf("------------------------------------- \n");
	}
    */
//************************************************************************************      
/*
    int *H_Full_CPU;
	H_Full_CPU               = (int *)malloc((L_A+1)*(L_B+1)*sizeof(int));
	float Start_CPU_Full_SW_Time =	cutGetTimerValue(hTimer);
	Full_SW_CPU(h_A, h_B, H_Full_CPU, L_A, L_B, si,dis, Gop, Gex);
	float End_CPU_Full_SW_Time =	cutGetTimerValue(hTimer);
	printf("CPU execution time:   %f ms\n", End_CPU_Full_SW_Time-Start_CPU_Full_SW_Time);

    for( unsigned int i = 0; i < (L_A+1)*(L_B+1); i++)
     {
     	 if (H_Full_CPU[i]>8)
     	 printf(" %i     %i      \n", i, H_Full_CPU[i]);
     } 
   	 printf("-------------------------------- \n");

					  

	if (MyProc == 0)
	{
		printf("GPU execution started...\n");
	}
*/	
	MPI_Barrier(MPI_COMM_WORLD);

	preallocBlockSums(L_A);
	preallocBlockSums_ss(L_M);


//------------------------------------------------------------------------- Kernel 1 START -------------------------------------------------------------------------------------------
    cudaEvent_t startH_Tilda, endH_Tilda, startTimeScan, endTimeScan, startH_Final, endH_Final, startH_Shrink, endH_Shrink, startCopyTime, endCopyTime, startBest200Match, endBest200Match, startStepTime, endStepTime;
    cudaEventCreate(&startH_Tilda); 
    cudaEventCreate(&endH_Tilda); 
    cudaEventCreate(&startTimeScan); 
    cudaEventCreate(&endTimeScan); 
    cudaEventCreate(&startH_Final); 
    cudaEventCreate(&endH_Final); 
    cudaEventCreate(&startH_Shrink);
    cudaEventCreate(&endH_Shrink);
    cudaEventCreate(&startCopyTime);
    cudaEventCreate(&endCopyTime);
    cudaEventCreate(&startBest200Match);
    cudaEventCreate(&endBest200Match);
    cudaEventCreate(&startStepTime);
    cudaEventCreate(&endStepTime);


	for (int i = 0; i < L_B; i++)
	{
        cudaEventRecord(startStepTime);
        cudaEventRecord(startH_Tilda);
		H_tilda<<<GridSize,256>>>(d_H, d_A, d_F, d_H_til,d_E_til,h_B[i], L_A,si,dis, Gop, Gex, Threads_N, 0);	
        cudaEventRecord(endH_Tilda);


        cudaEventRecord(startTimeScan);
		prescanArray(d_E_til, d_H_til,L_A);
		prescanArray(d_E_til, d_H_til,L_A);
        cudaEventRecord(endTimeScan);
	    
	    int Mimimum_Kernel1=h_Max_CPU[0];
        cudaEventRecord(startH_Final);
		Final_H<<<GridSize,256>>> (d_A, d_H_til, d_E_til, d_H, d_Max_H, d_Loc_H ,d_Con_Old, d_Con_New, h_B[i], L_A, L_M, Gop, Gex,Threads_N, Mimimum_Kernel1,0);
        cudaEventRecord(endH_Final);

        cudaEventRecord(startH_Shrink);
		sum_scan(d_Con_New,d_Con_Old,L_M);
		Shrink_H <<<GridSize,256>>>(d_Max_H2, d_Loc_H2, d_Max_H, d_Loc_H,d_Con_Old,d_Con_New, L_A, L_M, Threads_N, i);
        cudaEventRecord(endH_Shrink);

        cudaEventRecord(startCopyTime);
		CUDA_SAFE_CALL( cudaMemcpy(h_Con_New, d_Con_New + L_M - 1, sizeof(int), cudaMemcpyDeviceToHost) );
		int number=h_Con_New[0];
		int DATA_SZ_COPY = number * sizeof(int);
		CUDA_SAFE_CALL( cudaMemcpy(h_Max_GPU2, d_Max_H2  , DATA_SZ_COPY, cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaMemcpy(h_Loc_GPU2, d_Loc_H2,   DATA_SZ_COPY, cudaMemcpyDeviceToHost) );
        cudaEventRecord(endCopyTime);

		for( int qwe = 0; qwe < number; ++qwe) {	 if (h_Max_GPU2[qwe]>100) printf(" %i     %i       %i   %i   \n",i, qwe, h_Max_GPU2[qwe],h_Loc_GPU2[qwe]);	} 
    
        cudaEventRecord(startBest200Match);
		Kernel_1_Max_CPU(h_Max_GPU2,h_Max_CPU, h_A_Location, h_B_Location, h_Loc_GPU2, K1R,  number, i);  
        cudaEventRecord(endBest200Match);
        cudaEventRecord(endStepTime);
		
        cudaEventSynchronize(endH_Tilda); 
        cudaEventSynchronize(endTimeScan); 
        cudaEventSynchronize(endH_Final); 
        cudaEventSynchronize(endH_Shrink);
        cudaEventSynchronize(endCopyTime);
        cudaEventSynchronize(endBest200Match);
        cudaEventSynchronize(endStepTime);

        float H_Tilda_Time, Scan_Time, H_Final_Time, H_Shrink_Time, CPU_200_Time, Copy_Time, Step_Time;
        cudaEventElapsedTime(&H_Tilda_Time, startH_Tilda, endH_Tilda);
        cudaEventElapsedTime(&Scan_Time, startTimeScan, endTimeScan);
        cudaEventElapsedTime(&H_Final_Time, startH_Final, endH_Final);
        cudaEventElapsedTime(&H_Shrink_Time, startH_Shrink, endH_Shrink);
        cudaEventElapsedTime(&CPU_200_Time, startBest200Match, endBest200Match);
        cudaEventElapsedTime(&Copy_Time, startCopyTime, endCopyTime);
        cudaEventElapsedTime(&Step_Time, startStepTime, endStepTime);

		Total_Copy = Total_Copy + Copy_Time;
        /*
 	    printf(" %i  \n", i);
		printf("Time for H-Tilda:      %f ms,  %i percent\n", H_Tilda_Time, int((H_Tilda_Time/Step_Time)*100));
	    printf("Scan Time:             %f ms,  %i percent\n", Scan_Time   , int((Scan_Time/Step_Time)*100));
	    printf("Time for Final H:      %f ms,  %i percent\n", H_Final_Time, int((H_Final_Time/Step_Time)*100)); 	    
   	    printf("Time for Shrink H:     %f ms,  %i percent\n", H_Shrink_Time, int((H_Shrink_Time/Step_Time)*100)); 	    
	    printf("CPU 200 Best Match:    %f ms,  %i percent\n", CPU_200_Time, int((CPU_200_Time/Step_Time)*100)); 
   		printf("Copy Time from H-D-H:  %f ms,  %i percent\n", Copy_Time,int((Copy_Time/Step_Time)*100)); 
   																		   
	    printf("Time for Each Step:    %f ms\n", Step_Time); 
    	printf("---------------------------------------- \n");		                     
	    */
	}
//	printf("CPU or GPU ID:   %i, CPU Time:  %f ms,  GPU Time:  %f ms, CPU & GPU Time:  %f ms\n", MyProc, Total_CPU, Total_GPU, Total_Copy); 
	for (int i=0; i<K1R; i++)
		h_A_Location[i] +=Start_A;
// ---------------------------------  gathering Data form other PCs ---------------------------------------------------------	 
	
	int *h_Max_All_CPU, *h_A_Location_All_CPU, *h_B_Location_All_CPU;

	int DATA_SZ_MPI_K1   =  NumProcs*K1R*sizeof(int);
	int Number = NumProcs*K1R;

	h_Max_All_CPU        = (int *)malloc(DATA_SZ_MPI_K1);
	h_A_Location_All_CPU = (int *)malloc(DATA_SZ_MPI_K1);
	h_B_Location_All_CPU = (int *)malloc(DATA_SZ_MPI_K1);
    
    MPI_Barrier(MPI_COMM_WORLD);
    float Start_Kernel_1_Comm = cutGetTimerValue(hTimer);
	MPI_Gather( h_Max_CPU   , K1R, MPI_INT, h_Max_All_CPU       , K1R, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_A_Location, K1R, MPI_INT, h_A_Location_All_CPU, K1R, MPI_INT, 0, MPI_COMM_WORLD); 
	MPI_Gather( h_B_Location, K1R, MPI_INT, h_B_Location_All_CPU, K1R, MPI_INT, 0, MPI_COMM_WORLD); 
    float End_Kernel_1_Comm = cutGetTimerValue(hTimer);
	if (MyProc==0)
	{
		Kernel_1_Max_CPU_MPI(h_Max_All_CPU, h_A_Location_All_CPU, h_B_Location_All_CPU, 
							 h_Max_CPU_All, h_A_Location_All    , h_B_Location_All,
							 K1R          , Number);


        cudaEventRecord(Kernel1OverallTimeStop, 0);
        cudaEventSynchronize(Kernel1OverallTimeStop);
        float Kernel1TotalElapsedTime;
        cudaEventElapsedTime(&Kernel1TotalElapsedTime, Kernel1OverallTimeStart, Kernel1OverallTimeStop);

	//  printf("GPU execution finished...\n");
		//printf("CPU execution time for Kernel 1 :   %f ms \n", End_CPU_SW_Time-Start_CPU_SW_Time);
		printf("Total time spent in Kernel 1    :   %f ms \n", Kernel1TotalElapsedTime);
        printf("Host to Host Communication(P0)  :   %f\n",(End_Kernel_1_Comm - Start_Kernel_1_Comm)*100/Kernel1TotalElapsedTime);
   		printf("Copy Time from H-D-H            :   %f ms (%i percent of GPU Computation)\n", Total_Copy,int((Total_Copy/Kernel1TotalElapsedTime)*100)); 
		//printf("Speedup:                        :   %f    \n", (End_CPU_SW_Time-Start_CPU_SW_Time)/(End_Time_GPU-Start_Time_GPU));

		//unsigned int result_regtest_K1 = cutComparei( Max_CPU, h_Max_CPU_All, K1R);
		//unsigned int similarityReport_K1 = measureSimilarity(Max_CPU, h_Max_CPU_All, K1R, K1R);
        //Tolerance threshold is set to 95%
		//unsigned int result_regtest_K1   = similarityReport_K1 < (K1R  * 0.95)? 0:1;
        //printf("Kernel 1 Similarity             :   %f\n", (float)similarityReport_K1/(K1R));
		//printf("Verification of Kernel 1        :   %s    \n", (1 == result_regtest_K1) ? "PASSED" : "FAILED");

		printf("-------------------------------------------------------------\n");
	}	

    cudaEventDestroy(dataInitStart);
    cudaEventDestroy(dataInitStop);
	cudaEventDestroy(startH_Tilda); 
    cudaEventDestroy(endH_Tilda); 
    cudaEventDestroy(startTimeScan); 
    cudaEventDestroy(endTimeScan); 
    cudaEventDestroy(startH_Final); 
    cudaEventDestroy(endH_Final); 
    cudaEventDestroy(startH_Shrink);
    cudaEventDestroy(endH_Shrink);
    cudaEventDestroy(startCopyTime);
    cudaEventDestroy(endCopyTime);
    cudaEventDestroy(startBest200Match);
    cudaEventDestroy(endBest200Match);
    cudaEventDestroy(startStepTime);
    cudaEventDestroy(endStepTime);


	CUDA_SAFE_CALL(cudaFree(d_H));
	CUDA_SAFE_CALL(cudaFree(d_A));
    CUDA_SAFE_CALL(cudaFree(d_F));
    CUDA_SAFE_CALL(cudaFree(d_E_til));
    CUDA_SAFE_CALL(cudaFree(d_H_til));
    CUDA_SAFE_CALL(cudaFree(d_Max_H));
    CUDA_SAFE_CALL(cudaFree(d_Loc_H));
	CUDA_SAFE_CALL(cudaFree(d_Con_Old));
	CUDA_SAFE_CALL(cudaFree(d_Con_New));
 	CUDA_SAFE_CALL(cudaFree(d_Max_H2));
	CUDA_SAFE_CALL(cudaFree(d_Loc_H2));


	deallocBlockSums();    
	deallocBlockSums_ss();    
  

	free(h_H);
	free(h_Max_GPU);
	free(h_Loc_GPU);   
	free(h_Con_New);
	free(h_Loc_GPU2);
	free(h_Max_GPU2);
	free(Max_CPU);

}
