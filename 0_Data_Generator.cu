#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <iostream>
#include <fstream>
#include <assert.h>




void Data_Generator (int *h_A, int *h_B, 
					 int MyProc, int NumProcs, int L_B, int& L_A, int L_A1, int L_Gap, int& L_M, int Threads_N, int& Start_A, int GridSize, int BlockSize,
					 int& DATA_SZ_A, int& DATA_SZ_H,  int& DATA_SZ_M, int& DATA_SZ_L,
                     int argc, char** argv)
{
    int Length = (L_A1-2*L_Gap)/NumProcs;
    L_A = Length + 2*L_Gap;
    MPI_Status status;
    if (MyProc == 0)
    {
        printf("Input Data ...\n");
        printf("Number of Processors is    %i \n", NumProcs);
        printf("Length of Sequence of A is %i \n", L_A1);
        printf("Length of Sequence of B is %i \n", L_B);
        //		printf("Block Per Grid is          %i \n", GridSize);
        //		printf("Threads Per Block is       %i \n", BlockSize);
        //printf("Maximum GPU Memory is      %i MB \n", int((10.5*((L_A1-2*L_Gap)/NumProcs+2*L_Gap)/1024. + L_B/1024.)/1024.0*sizeof(int)));
        printf("-------------------------------------------------------------\n");
        printf("Initializing data...\n");
        printf("...Generating input data in CPU memory\n");

#ifndef FILEREAD
        for(int i = 0; i < L_A1; i++)
        {
            h_A[i] = int ((rand() % 20)+1);
        }

        for(int i = 0; i < L_B; i++)
        {
            h_B[i] = (rand() % 20)+1;
        } 
#endif
#ifdef FILEREAD
        //Making use of public repository for the alignment
        FILE *fp; int c, count;
        
        //Main sequence
        count = 0;
        if (argc != 5) {
            fprintf(stderr, "Usage: %s Main_Size Query_Size Main_file.txt Query_file.txt\n", argv[0]);
            exit(1);
        }
        if (!(fp = fopen(argv[3], "r"))) {
            perror(argv[3]);
            exit(1);
        }
        while (count < L_A1) 
        {
            c = fgetc(fp);
            assert(c != EOF  /* Main sequeunce file is too shorter than the length specified */);
            if(isalpha(c))
            {
                count++;
                h_A[count] = c;
            }
        }
        fclose(fp);

        //Query sequence
        count = 0;
        if (!(fp = fopen(argv[4], "r"))) {
            perror(argv[4]);
            exit(1);
        } 
        while (count < L_B) 
        { 
            c = fgetc(fp);
            assert(c != EOF  /* Query sequeunce file is too shorter than the length specified */);
            if(isalpha(c))
            {
                count++;
                h_B[count] = c;
            }
        }
        fclose(fp);
#endif
    }

	MPI_Bcast( h_B, L_B , MPI_INT, 0, MPI_COMM_WORLD);
    if (MyProc == 0)
	{
		for (int i=1; i<NumProcs; i++)
		{
			MPI_Send( h_A + i*Length, L_A, MPI_INT, i, 111, MPI_COMM_WORLD );
		}
	}
	else
	{
		MPI_Recv( h_A     , L_A, MPI_INT, 0, 111, MPI_COMM_WORLD, &status );		
	}

    FILE *f = fopen("/tmp/Database.in","w");
    if (f==NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    // Writing DB Sequence to file
    fwrite(h_A, sizeof(int) , L_A, f);
    fclose(f);

    f = fopen("/tmp/Query.in","w");
    if (f==NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    // Writing Query Sequence to file
    fwrite(h_B, sizeof(int) , L_B, f);
    fclose(f);
}
void Data_Fetch (int *h_A, int *h_B, 
					 int MyProc, int NumProcs, int L_B, int& L_A, int L_A1, int L_Gap, int& L_M, int Threads_N, int& Start_A, int GridSize, int BlockSize,
					 int& DATA_SZ_A, int& DATA_SZ_H,  int& DATA_SZ_M, int& DATA_SZ_L)
{

     unsigned int hTimer;

	CUT_SAFE_CALL ( cutCreateTimer(&hTimer) );
    CUDA_SAFE_CALL( cudaThreadSynchronize() );
    CUT_SAFE_CALL ( cutResetTimer(hTimer)   );
    CUT_SAFE_CALL ( cutStartTimer(hTimer)   );

     //Assuming that we start our algorithm by reading the file
    float masterFileReadingStart = cutGetTimerValue(hTimer);
    int Length = (L_A1-2*L_Gap)/NumProcs;
    L_A = Length + 2*L_Gap;
    Start_A = MyProc*Length;

    //Read Query Sequence from file
    FILE *f = fopen("/tmp/Query.in","r");
    if (f==NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(h_B, sizeof(int), L_B, f);
    fclose(f);

    //Read DB Sequence from file
    f = fopen("/tmp/Database.in","r");
    if (f==NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }
    fread(h_A, sizeof(int), L_A, f);
    fclose(f);

    MPI_Barrier(MPI_COMM_WORLD);
    float masterFileReadingEnd = cutGetTimerValue(hTimer);
    float masterFileReadingTime = masterFileReadingEnd - masterFileReadingStart;
    float masterFileReadingTimeMax;
    MPI_Reduce(&masterFileReadingTime, &masterFileReadingTimeMax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
    if (MyProc==0)
        printf("Sequence Sharing done in        :   %f ms \n", masterFileReadingTimeMax);

	DATA_SZ_A  =  L_A            * sizeof(int);
	DATA_SZ_H  = 2*(L_A+1)       * sizeof(int);

	if (L_A>Threads_N)
	{
		L_M = Threads_N + 1;
		DATA_SZ_L = (int ((L_A-1)/Threads_N+1))*Threads_N*sizeof(int); 
	}
	else
	{
		L_M = L_A +1 ; 
	    DATA_SZ_L =DATA_SZ_A; 
	}
	DATA_SZ_M = L_M * sizeof(int);
	
}





//	ofstream tests;
//	tests.open ("times.txt");
