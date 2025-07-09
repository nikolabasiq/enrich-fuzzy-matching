from libcpp.vector cimport vector

cimport cython
cimport numpy as np
import numpy as np


np.import_array()


ctypedef fused  float_ft:
	cython.float
	cython.double


cdef extern from "topn_parallel.h":

	cdef int topn_parallel[T](
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_rows,
		int n_jobs
	) except +;

	cdef void sparse_topn_parallel[T](
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],  # data of A
		int ntop,
		int Bp[],
		int Bj[],
		T Bx[],  # data of output
		int n_jobs
	) except +;

	cdef void sparse_hstack_parallel[T](
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],  # data of A
		int Bp[],
		int Bj[],
		T Bx[],  # data of output
		int n_jobs
	) except +;

cpdef topn_threaded(
	np.ndarray[int, ndim=1] r,
	np.ndarray[int, ndim=1] c,
	np.ndarray[float_ft, ndim=1] d,
	int ntop,
	int n_rows,
	int n_jobs
):
	"""
	Cython glue function:
	r, c, and d are 1D numpy arrays all of the same length N. 
	This function will return arrays rn, cn, and dn of length n <= N such
	that the set of triples {(rn[i], cn[i], dn[i]) : 0 < i < n} is a subset of 
	{(r[j], c[j], d[j]) : 0 < j < N} and that for every distinct value 
	x = rn[i], dn[i] is among the first ntop existing largest d[j]'s whose 
	r[j] = x.

	Input:
		r and c: two 1D integer arrays of the same length
		d: 1D array of single or double precision floating point type of the
		same length as r or c
		ntop maximum number of maximum d's returned
		use_threads: use multi-thread or not
		n_jobs: number of threads, must be >= 1

	Output:
		(rn, cn, dn) where rn, cn, dn are all arrays as described above.
	"""

	cdef int* var_r = &r[0]
	cdef int* var_c = &c[0]
	cdef float_ft* var_d = &d[0]
	
	n = len(c)
	
	new_len = topn_parallel(
		n, var_r, var_c, var_d, ntop, n_rows, n_jobs
	)
	
	
	return new_len

cpdef sparse_topn_threaded(
	int n_blocks,
	int n_row,
	np.ndarray[int, ndim=1] n_cols,
	np.ndarray[int, ndim=1] a_indptr,
	np.ndarray[int, ndim=1] a_indices,
	np.ndarray[float_ft, ndim=1] a_data,
	int ntop,
	np.ndarray[int, ndim=1] b_indptr,
	np.ndarray[int, ndim=1] b_indices,
	np.ndarray[float_ft, ndim=1] b_data,
	int n_jobs
):

	cdef int* n_col = &n_cols[0]
	cdef int* Ap = &a_indptr[0]
	cdef int* Aj = &a_indices[0]
	cdef float_ft* Ax = &a_data[0]
	cdef int* Bp = &b_indptr[0]
	cdef int* Bj = &b_indices[0]
	cdef float_ft* Bx = &b_data[0]
	
	sparse_topn_parallel(n_blocks, n_row, n_col, Ap, Aj, Ax, ntop, Bp, Bj, Bx, n_jobs)

cpdef sparse_hstack_threaded(
	int n_blocks,
	int n_row,
	np.ndarray[int, ndim=1] n_cols,
	np.ndarray[int, ndim=1] a_indptr,
	np.ndarray[int, ndim=1] a_indices,
	np.ndarray[float_ft, ndim=1] a_data,
	np.ndarray[int, ndim=1] b_indptr,
	np.ndarray[int, ndim=1] b_indices,
	np.ndarray[float_ft, ndim=1] b_data,
	int n_jobs
):

	cdef int* n_col = &n_cols[0]
	cdef int* Ap = &a_indptr[0]
	cdef int* Aj = &a_indices[0]
	cdef float_ft* Ax = &a_data[0]
	cdef int* Bp = &b_indptr[0]
	cdef int* Bj = &b_indices[0]
	cdef float_ft* Bx = &b_data[0]
	
	sparse_hstack_parallel(n_blocks, n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx, n_jobs)

