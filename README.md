# Compare CUDA's SVD libraries in Python

There are two different SVD (Single Value Decomposition) libraries for GPU, currently.

This code compares the following functions. (both single and double presicion floating point)

- CPU：NumPy (numpy.linalg.svd)
- GPU：cuSOLVER ([cusolverDnSgesvd](http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-gesvd) and cusolverDnDgesvd)
- GPU：MAGMA ([magma_sgesvd](http://icl.cs.utk.edu/projectsfiles/magma/doxygen/group__magma__gesvd.html) and magma_dgesvd)

This code also tests options for computation (e.g. A,S,O and N).

- 'full_matrices=Falase' for NumPy
- 'A', 'S' and 'N' for cuSOLVER
- 'A' and 'S' for MAGMA

# Requirements

- Pythone 2.7.11
- [scikit-cuda 0.5.2](https://github.com/lebedov/scikit-cuda/tree/master/skcuda)
- [NumPy 1.10.4](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html)
- [MAGMA 2.0.2](http://icl.cs.utk.edu/magma/software/browse.html?start=0&per=5)
- [CUDA 8.0](http://docs.nvidia.com/cuda/cusolver/#cuds-lt-t-gt-gesvd)

# Usage

```Shell
% python ./svd_on_CPU_GPU.py [col_size] [row_size]
```

## Example

Compute and compare double-presicion floating point (float64) 1000x1000 array.

```Shell
% python ./svd_on_CPU_GPU.py 1000 1000

Single Value Decomposition (SVD) on CPU and GPU
input array size:  1000 x 1000 ( float64 )

CPU: numpy.linalg.svd()
np.linalg.svd():  0.994359016418 [sec]
check result
Solution is ... True
cpu_numpy_linalg_svd 1.02781414986

GPU: skcuda.linalg.svd() internaly use cusolver
H2D:  0.00232005119324 [sec]
skcuda.linalg.svd():  1.49890089035 [sec]
Total:  1.50123405457 [sec]
check result
Solution is ... True

GPU: skcuda.cusolver.cusolverDnDgesvd() 'A' option
H2D:  0.00201988220215 [sec]
solver.cusolverDnSgesvd() 'A' option:  0.298828125 [sec]
Total:  0.672796964645 [sec]
check result
Solution is ... True

GPU: skcuda.cusolver.cusolverDnDgesvd() 'S' option
H2D:  0.0031840801239 [sec]
solver.cusolverDnSgesvd() 'S' option 0.283460140228 [sec]
Total:  0.289812088013 [sec]
check result
Solution is ... True

GPU: skcuda.cusolver.cusolverDnDgesvd() 'N' option
H2D:  0.0019850730896 [sec]
solver.cusolverDnSgesvd() 'N' option:  0.208176136017 [sec]
Total:  0.211440086365 [sec]
only s is computed

GPU: skcuda.magma_gesvd 'A' option
magma_sgesvd A option (= Total):  0.485193967819 [sec]
Total:  0.485193967819 [sec]
check result
Solution is ... True

GPU: skcuda.magma_gesvd 'S' option
magma_sgesvd S option (= Total):  0.659060955048 [sec] 694
Total:  0.659060955048 [sec]
check result
Solution is ... True
gpu_magma_magma_gesvd_S 0.690434932709
main 5.18893098831
```
