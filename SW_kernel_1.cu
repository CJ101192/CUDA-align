#include "../include/cudaPrint/cuPrintf.cu"
__global__ void diagonalComputation(int mainSequenceLength, int querySequenceLength, int *mainSeqAll, int *querySeq, 
        int  *maxScoresPerBlock,
        int Length,
        int matchScore, int mismatchScore, int gapStartScore, int gapExtensionScore)
{  
    int hVal = 0, eVal = 0, fVal = 0;               //Score of current cell being handled by this thread
    extern __shared__ int buffer[];

    //H_score_A  -- for main scores of previous of previous diagonal
    //H_score_B  -- for scores of previous diagonal

    int myRow = threadIdx.x;
    int myBlock = blockIdx.x;

    int *mainSeq = &mainSeqAll[myBlock * Length];                       //This block should compute a portion of whole mainSequence
    
    //Postion following buffers through shared memory
    int *maxScores = &buffer[0*blockDim.x];         //Score keeper for threads of this Block
    int *F_score = &buffer[1*blockDim.x];
    int *H_score_A = &buffer[2*blockDim.x];
    int *H_score_B = &buffer[3*blockDim.x];

    maxScores[myRow] = 0;

    H_score_B[myRow] = 0;
    
    __syncthreads();

    //Begin computing scores 
    int tidPos = 0;     //current column position of tid
    int myQueryChar = querySeq[myRow];
    for (int i=1; i<= mainSequenceLength + querySequenceLength -1; i++)
    {
        if (tidPos + myRow < i && tidPos < mainSequenceLength)    //Should this thread compute?
        { 
            //Horizontal dependency
            eVal = max(eVal - gapExtensionScore, H_score_B[myRow] - gapStartScore);

            //Vertical Dependency
            int upperFValue = (myRow==0)?  0 : F_score[myRow-1];
            int upperHValue = (myRow==0)?  0 : H_score_B[myRow -1];
            fVal = max(upperFValue - gapExtensionScore, upperHValue - gapStartScore);

            //Diagonal Dependency
            int simScore = (mainSeq[tidPos] == myQueryChar)? matchScore : mismatchScore;
            int diagonalHValue = (myRow==0)?  0 : H_score_A[myRow -1];
            hVal = diagonalHValue + simScore;
            
            //Maxima
            hVal = max(max(eVal, fVal), max(hVal, 0));

            tidPos++;
        }


        __syncthreads();        //To make sure this diagonal is computed
        //Save the values
        H_score_A[myRow] = H_score_B[myRow];

        __syncthreads();        //To make sure A is saved before editing B

        F_score[myRow] = fVal;
        H_score_B[myRow] = hVal;
        maxScores[myRow] = max(maxScores[myRow], hVal);

        __syncthreads();        //To make sure values are saved before the computation of next diagonal
    }

    //Save the maximum score of this block
    if(myRow==0)
    {
        maxScoresPerBlock[myBlock] = 0;
        for (int i=0; i< querySequenceLength; i++)
            maxScoresPerBlock[myBlock] = max( maxScoresPerBlock[myBlock], maxScores[i]);
    }
}
