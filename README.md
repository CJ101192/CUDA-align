CUDA-align
==========

Efficient CUDA code for pairwise local alignment of two sequences.
Use makefile to compile the code.
To execute type : mpirun SW_exe argument1 argument2

argument1: size of the subject sequence,

argument2: size of the query sequence

The algorithm implemented by this codebase is described in the following peer-reviewed publication. Please cite this paper, when using our code for academic purposes:
> **Chirag Jain, Subodh Kumar.** "Fine-grained GPU parallelization of pairwise local sequence alignment" *High Performance Computing (HiPC), 2014 21st International Conference on. IEEE, 2014*. [![dx.doi.org/10.1109/HiPC.2014.7116912](https://img.shields.io/badge/doi-10.1109%2FHiPC.2014.7116912-blue.svg)](http://dx.doi.org/10.1109/HiPC.2014.7116912)
