from libcpp.vector cimport vector

cimport cython
cimport numpy as np
import numpy as np


np.import_array()


ctypedef fused  float_ft:
	cython.float
	cython.double


cdef extern from "topn_source.h":

	cdef void sparse_topn_source[T](
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],  # data of A
		int ntop,
		int Bp[],
		int Bj[],
		T Bx[]  # data of output
	) except +;

	cdef void sparse_hstack_source[T](
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],  # data of A
		int Bp[],
		int Bj[],
		T Bx[]  # data of output
	) except +;


cpdef sparse_topn(
	int n_blocks,
	int n_row,
	np.ndarray[int, ndim=1] n_cols,
	np.ndarray[int, ndim=1] a_indptr,
	np.ndarray[int, ndim=1] a_indices,
	np.ndarray[float_ft, ndim=1] a_data,
	int ntop,
	np.ndarray[int, ndim=1] b_indptr,
	np.ndarray[int, ndim=1] b_indices,
	np.ndarray[float_ft, ndim=1] b_data
):
	"""
	Cython glue function to call sparse_topn_source C++
	implementation.  This function will return the same
	CSR matrix A it is given but with only the top ntop
	elements of each row sorted in descending order.

	Input:
		n_row: number of rows of A matrix
		n_col: number of columns of B matrix

		Ap, Aj, Ax: CSR expression of A matrix

		ntop: n top results

	Output by reference:
		Ap, Aj, Ax: CSR expression of A matrix
	
	N.B. A must be CSR format!!!
	"""

	cdef int* n_col = &n_cols[0]
	cdef int* Ap = &a_indptr[0]
	cdef int* Aj = &a_indices[0]
	cdef float_ft* Ax = &a_data[0]
	cdef int* Bp = &b_indptr[0]
	cdef int* Bj = &b_indices[0]
	cdef float_ft* Bx = &b_data[0]
	
	sparse_topn_source(n_blocks, n_row, n_col, Ap, Aj, Ax, ntop, Bp, Bj, Bx)


cpdef sparse_hstack(
	int n_blocks,
	int n_row,
	np.ndarray[int, ndim=1] n_cols,
	np.ndarray[int, ndim=1] a_indptr,
	np.ndarray[int, ndim=1] a_indices,
	np.ndarray[float_ft, ndim=1] a_data,
	np.ndarray[int, ndim=1] b_indptr,
	np.ndarray[int, ndim=1] b_indices,
	np.ndarray[float_ft, ndim=1] b_data
):
	"""
	Cython glue function to call sparse_topn_source C++
	implementation.  This function will return the same
	CSR matrix A it is given but with only the top ntop
	elements of each row sorted in descending order.

	Input:
		n_row: number of rows of A matrix
		n_col: number of columns of B matrix

		Ap, Aj, Ax: CSR expression of A matrix

	Output by reference:
		Ap, Aj, Ax: CSR expression of A matrix
	
	N.B. A must be CSR format!!!
	"""

	cdef int* n_col = &n_cols[0]
	cdef int* Ap = &a_indptr[0]
	cdef int* Aj = &a_indices[0]
	cdef float_ft* Ax = &a_data[0]
	cdef int* Bp = &b_indptr[0]
	cdef int* Bj = &b_indices[0]
	cdef float_ft* Bx = &b_data[0]
	
	sparse_hstack_source(n_blocks, n_row, n_col, Ap, Aj, Ax, Bp, Bj, Bx)
