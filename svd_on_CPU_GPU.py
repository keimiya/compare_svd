# -* coding: utf-8 -*-

import sys
import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver
import skcuda.linalg
import skcuda.cusolver as solver
import skcuda.magma as magma
import time
from functools import wraps

def stop_watch(func):
        @wraps(func)
        def wrapper(*args, **kargs):
                start = time.time()
                result = func(*args, **kargs)
                elapsed_time = time.time() - start
                print func.__name__, elapsed_time
                return result
        return wrapper

def check_result(input, u,s,v):
        print "check result"        
        # print "U=\n", u
        # print "S=\n", s
        # print "V=\n", v
        # print 'Solution is ...', np.allclose(input, np.dot(u, np.dot(np.diag(s), v)), 1e-4)
        # print np.dot(U, np.dot(np.diag(S), V))

@stop_watch        
def cpu_numpy_linalg_svd(input):
        coloring_print("\nCPU: numpy.linalg.svd()")
	cpu_numpy_linalg_svd_start  = time.time()
	u, s, v = np.linalg.svd(input, full_matrices=False)
        cpu_numpy_linalg_svd_end = time.time()
	print "np.linalg.svd(): ", cpu_numpy_linalg_svd_end - cpu_numpy_linalg_svd_start, "[sec]"

        check_result(input,u,s,v)

        
def gpu_skcuda_linalg_svd(input):
        coloring_print("\nGPU: skcuda.linalg.svd() internaly use cusolver")

        h2d_start = time.time()
        input_gpu = gpuarray.to_gpu(input)
        h2d_end   = time.time()
        print "H2D: ", h2d_end - h2d_start, "[sec]"

        gpu_svd_start        = time.time()
        skcuda.linalg.init()
        u_gpu, s_gpu, vh_gpu = skcuda.linalg.svd(input_gpu, 'A', 'A', 'cusolver') # use sciki-cuda ver 0.5.2
        gpu_svd_end          = time.time()
        print "skcuda.linalg.svd(): ", gpu_svd_end - gpu_svd_start, "[sec]"
        print "Total: ", gpu_svd_end - h2d_start,  "[sec]"

        # u and s is swapped (数学的に正しいかはわからない)
        check_result(input, u_gpu.get(), s_gpu.get(), vh_gpu.get())

        
def gpu_skcuda_cusolver_cusolverDnDgesvd_A(input):
        coloring_print("\nGPU: skcuda.cusolver.cusolverDnDgesvd() 'A' option")

        # #縦横を逆(≒転地)してcolumn-majorにし、GPUのcusolverDnDgesvd()に対応させる。結果配列のU,Vが逆になる
        n, m = input.shape

        # change function by data type
        if input.dtype == np.dtype('float64'):
                get_buffer  = solver.cusolverDnDgesvd_bufferSize
                cusolver_svd = solver.cusolverDnDgesvd
        elif input.dtype == np.dtype('float32'):
                get_buffer  = solver.cusolverDnSgesvd_bufferSize
                cusolver_svd = solver.cusolverDnSgesvd
        else:
                print "Error: data type must be float64 or float32"
        
        h2d_start = time.time()
        input_gpu = gpuarray.to_gpu(input)
        h2d_end   = time.time()
        print "H2D: ", h2d_end - h2d_start, "[sec]"

        # Set up work buffers:
        h             = solver.cusolverDnCreate()
        Lwork         = get_buffer(h, m, n)
        workspace_gpu = gpuarray.zeros(Lwork, input.dtype)
        devInfo_gpu   = gpuarray.zeros(1, np.int32)
        
        # Set up output buffers:
        s_gpu  = gpuarray.zeros(min(m, n), input.dtype)        
        u_gpu  = gpuarray.zeros((m, m), input.dtype)
        vh_gpu = gpuarray.zeros((n, n), input.dtype)
        
        # Compute:
        # 'A': all m columns of U are returned in array U
	cusolver_1_svd_start  = time.time()
        status = cusolver_svd(h, 'A', 'A',
        		     m, n, input_gpu.gpudata, m, s_gpu.gpudata,
                             u_gpu.gpudata, m,
        		     vh_gpu.gpudata, n,
                             workspace_gpu.gpudata,
        		     Lwork,
        		     0,
        		     devInfo_gpu.gpudata)
        cusolver_1_svd_end = time.time()
	print "solver.cusolverDnSgesvd() 'A' option: ", cusolver_1_svd_end - cusolver_1_svd_start, "[sec]"
        print "Total: ", cusolver_1_svd_end - h2d_start,  "[sec]"        

        # u and s is swapped (数学的に正しいかはわからない)        
        check_result(input, vh_gpu.get(), s_gpu.get(), u_gpu.get())

        
        
def gpu_skcuda_cusolver_cusolverDnDgesvd_S(input):
        coloring_print("\nGPU: skcuda.cusolver.cusolverDnDgesvd() 'S' option")

        # #縦横を逆(≒転地)してcolumn-majorにし、GPUのcusolverDnDgesvd()に対応させる。結果配列のU,Vが逆になる
        n, m = input.shape

        # change function by data type
        if input.dtype == np.dtype('float64'):
                get_buffer  = solver.cusolverDnDgesvd_bufferSize
                cusolver_svd = solver.cusolverDnDgesvd
        elif input.dtype == np.dtype('float32'):
                get_buffer  = solver.cusolverDnSgesvd_bufferSize
                cusolver_svd = solver.cusolverDnSgesvd
        else:
                print "Error: data type must be float64 or float32"
        
        h2d_start = time.time()
        input_gpu = gpuarray.to_gpu(input)
        h2d_end   = time.time()
        print "H2D: ", h2d_end - h2d_start, "[sec]"
        
        # Set up work buffers:
        h             = solver.cusolverDnCreate()
        Lwork         = get_buffer(h, m, n)
        workspace_gpu = gpuarray.zeros(Lwork, input.dtype)
        devInfo_gpu   = gpuarray.zeros(1, np.int32)
       
        # Set up output buffers:
        s_gpu  = gpuarray.zeros(min(m, n), input.dtype)        
        u_gpu  = gpuarray.zeros((n, n), input.dtype)
        vh_gpu = gpuarray.zeros((m, m), input.dtype)
        
        # 'S': the first min(m,n) columns of U (the left singular vectors) are returned in the array U
	cusolver_S_svd_start  = time.time()        
        status = cusolver_svd(h, 'S', 'S',
        		     m, n, input_gpu.gpudata, m, s_gpu.gpudata,
                             u_gpu.gpudata, m,
        		     vh_gpu.gpudata, n,
                             workspace_gpu.gpudata, Lwork, 0, devInfo_gpu.gpudata)
        cusolver_S_svd_end = time.time()
      
	print "solver.cusolverDnSgesvd() 'S' option", cusolver_S_svd_end - cusolver_S_svd_start, "[sec]"
        print "Total: ", cusolver_S_svd_end - h2d_start,  "[sec]"

        # u and s is swapped (数学的に正しいかはわからない)                
        check_result(input, vh_gpu.get(), s_gpu.get(), u_gpu.get())
        
        solver.cusolverDnDestroy(h)



def gpu_skcuda_cusolver_cusolverDnDgesvd_N(input):
        coloring_print("\nGPU: skcuda.cusolver.cusolverDnDgesvd() 'N' option")

        # #縦横を逆(≒転地)してcolumn-majorにし、GPUのcusolverDnDgesvd()に対応させる。結果配列のU,Vが逆になる
        n, m = input.shape

        # change function by data type
        if input.dtype == np.dtype('float64'):
                get_buffer  = solver.cusolverDnDgesvd_bufferSize
                cusolver_svd = solver.cusolverDnDgesvd
        elif input.dtype == np.dtype('float32'):
                get_buffer  = solver.cusolverDnSgesvd_bufferSize
                cusolver_svd = solver.cusolverDnSgesvd
        else:
                print "Error: data type must be float64 or float32"
        
        h2d_start = time.time()
        input_gpu = gpuarray.to_gpu(input)
        h2d_end   = time.time()
        print "H2D: ", h2d_end - h2d_start, "[sec]"
        
        # Set up work buffers:
        h             = solver.cusolverDnCreate()
        Lwork         = get_buffer(h, m, n)
        workspace_gpu = gpuarray.zeros(Lwork, input.dtype)
        devInfo_gpu   = gpuarray.zeros(1, np.int32)
        
        # Set up output buffers:
        s_gpu  = gpuarray.zeros(min(m, n), input.dtype)        
        
        # 'N': no columns of U (no left singular vectors) are computed.
	cusolver_N_svd_start  = time.time()                
        status = cusolver_svd(h, 'N', 'N',
        		     m, n, input_gpu.gpudata, m, s_gpu.gpudata,
                             0, m,
        		     0, n,
                             workspace_gpu.gpudata, 0, 0, devInfo_gpu.gpudata)
        cusolver_N_svd_end = time.time()
	print "solver.cusolverDnSgesvd() 'N' option: ", cusolver_N_svd_end - cusolver_N_svd_start, "[sec]"
        print "Total: ", cusolver_N_svd_end - h2d_start,  "[sec]"        

        print "only s is computed"
        # print s_gpu.get()
        
        solver.cusolverDnDestroy(h)


def gpu_magma_magma_gesvd_A(input):        
	coloring_print("\nGPU: skcuda.magma_gesvd 'A' option")
        input_original = input.copy() # magma is descructive
        magma.magma_init()

        # #縦横を逆(≒転地)してcolumn-majorにし、GPUのcusolverDnDgesvd()に対応させる。結果配列のU,Vが逆になる
        n, m = input.shape
        
        # change function by data type
        if input.dtype == np.dtype('float64'):
                get_buffer = magma.magma_dgesvd_buffersize
                magma_svd  = magma.magma_dgesvd
        elif input.dtype == np.dtype('float32'):
                get_buffer = magma.magma_sgesvd_buffersize
                magma_svd  = magma.magma_sgesvd
        else:
                print "Error: data type must be float64 or float32"

        # Set up work buffers:                
        Lwork     = get_buffer('A', 'A', m, n)
        workspace = np.zeros(Lwork, input.dtype)
        
        # Set up output buffers:        
        s         = np.zeros(min(m, n), input.dtype)
        # u         = np.zeros((n, n), input.dtype)
        # vh        = np.zeros((m, m), input.dtype)
        u         = np.zeros((m, m), input.dtype)
        vh        = np.zeros((n, n), input.dtype)

        # Compute:
        magma_svd_A_start  = time.time()
	status = magma_svd('A',       # jobu
			   'A',                   # jobvt
			   m,                     # m
			   n,                     # n
			   input.ctypes.data,     # A
			   m,                     # lda
			   s.ctypes.data,         # s
			   u.ctypes.data,         # U
			   m,                     # ldu
			   vh.ctypes.data,        # VT
			   n,                     # ldvt
			   workspace.ctypes.data, # work
			   Lwork)                 # lwork
        magma_svd_A_end = time.time()
	print "magma_sgesvd A option (= Total): ", magma_svd_A_end - magma_svd_A_start, "[sec]"
        print "Total: ", magma_svd_A_end - magma_svd_A_start, "[sec]"


        # magma is descructive, u and s is swapped (数学的に正しいかはわからない)                        
        check_result(input_original, vh, s, u)

        magma.magma_finalize()

@stop_watch
def gpu_magma_magma_gesvd_S(input):
	coloring_print("\nGPU: skcuda.magma_gesvd 'S' option")
        input_original = input.copy()
        magma.magma_init()

        # #縦横を逆(≒転地)してcolumn-majorにし、GPUのcusolverDnDgesvd()に対応させる。結果配列のU,Vが逆になる
        n, m = input.shape

        # change function by data type
        if input.dtype == np.dtype('float64'):
                get_buffer = magma.magma_dgesvd_buffersize
                magma_svd  = magma.magma_dgesvd
        elif input.dtype == np.dtype('float32'):
                get_buffer = magma.magma_sgesvd_buffersize
                magma_svd  = magma.magma_sgesvd
        else:
                print "Error: data type must be float64 or float32"

        Lwork     = get_buffer('A', 'A', m, n)
        workspace = np.zeros(Lwork, input.dtype)
        # The size of u and vh in 'S' are different from those of 'A'
	s  = np.zeros(min(m, n), input.dtype)
	u  = np.zeros((n, m), input.dtype)
	vh = np.zeros((n, n), input.dtype)
                
        # Compute:
        magma_svd_start  = time.time()
	status = magma_svd('S',       # jobu
			   'S',                   # jobvt
			   m,                     # m
			   n,                     # n
			   input.ctypes.data,     # A
			   m,                     # lda
			   s.ctypes.data,         # s
			   u.ctypes.data,         # U
			   m,                     # ldu
			   vh.ctypes.data,        # VT
			   n,                     # ldvt
			   workspace.ctypes.data, # work
			   Lwork)                 # lwork
        magma_svd_end = time.time()
	print "magma_sgesvd S option (= Total): ", magma_svd_end - magma_svd_start, "[sec]", (pycuda.driver.mem_get_info()[1] - pycuda.driver.mem_get_info()[0])/1024/1024
        print "Total: ", magma_svd_end - magma_svd_start, "[sec]"

        # magma is descructive, u and s is swapped (数学的に正しいかはわからない)                                
        check_result(input_original, vh, s, u)

        magma.magma_finalize()

        

def coloring_print(string):
        print "\033[92m",string,"\033[0m"

#@profile # for line_profiler.py
@stop_watch
def main():
        coloring_print("Single Value Decomposition (SVD) on CPU and GPU")
        #input = np.random.randn(63001, 1999) # actual size C_CONTIGUOUS( = row major)な縦長配列
	col = int(sys.argv[1])
	row = int(sys.argv[2])
	input = np.ones([col,row]) * 0.5
	input = np.asarray(input, np.float64)

	print "input array size: ", input.shape[0], "x", input.shape[1], "(", input.dtype, ")"

        cpu_numpy_linalg_svd(input)
        # gpu_skcuda_linalg_svd(input) 
        # gpu_skcuda_cusolver_cusolverDnDgesvd_A(input)
        # gpu_skcuda_cusolver_cusolverDnDgesvd_S(input)
        # gpu_skcuda_cusolver_cusolverDnDgesvd_N(input)
        # gpu_magma_magma_gesvd_A(input)
        gpu_magma_magma_gesvd_S(input)



if __name__ == '__main__':
        main()
