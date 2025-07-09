#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>

#include "./topn_source.h"

/*
	C++ implementation of sparse_topn_source

	This function will return the same CSR matrix A it is given but with only the
	top ntop elements of each row sorted in descending order.

	Input:
		n_row: number of rows of A matrix
		n_col: number of columns of B matrix

		Ap, Aj, Ax: CSR expression of A matrix

		ntop: n top results

	Output by reference:
		Ap, Aj, Ax: CSR expression of A matrix
	N.B. A must be CSR format!!!
*/
template<typename T>
void sparse_topn_source(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int ntop,
		int Bp[],
		int Bj[],
		T Bx[]	//data of output
)
{
	std::vector<int> col_offsets(n_blocks + 1);
	col_offsets[0] = 0;
	std::partial_sum(n_col, n_col + n_blocks, col_offsets.data() + 1);
	int n_col_sum = col_offsets[n_blocks];

	std::vector<int*> bAp(n_blocks);
	std::vector<int*> bAj(n_blocks);
	std::vector<T*> bAx(n_blocks);
	bAp[0] = Ap;
	bAj[0] = Aj;
	bAx[0] = Ax;
	for (int b = 1; b < n_blocks; b++){
		bAp[b] = bAp[b - 1] + (n_row + 1);
		bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
		bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
	}
	std::vector<Candidate<T>> candidates;
	candidates.reserve(n_col_sum);
	Bp[0] = 0;
	int s = 0;
	for(int i = 0; i < n_row; i++){
		for (int b = 0; b < n_blocks; b++){
			int jj_start = bAp[b][i];
			int jj_end = bAp[b][i + 1];
			for(int jj = jj_start; jj < jj_end; jj++){
				Candidate<T> c;
				c.index = bAj[b][jj] + col_offsets[b];
				c.value = bAx[b][jj];
				candidates.push_back(c);
			}
		}

		int len = (int) candidates.size();
		if (len > ntop){
			std::partial_sort(candidates.data(), candidates.data() + ntop, candidates.data() + len);
			len = ntop;
		}
		else{
			std::sort(candidates.data(), candidates.data() + len);
		}
		for(int jj = 0; jj < len; jj++){
			Candidate<T> c = candidates[jj];
			Bj[s] = c.index;
			Bx[s++] = c.value;
		}
		Bp[i + 1] = s;
		candidates.clear();
	}
}
template void sparse_topn_source<float>(int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], float Ax[], int ntop, int Bp[], int Bj[], float Bx[]);
template void sparse_topn_source<double>(int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], double Ax[], int ntop, int Bp[], int Bj[], double Bx[]);

/*
	C++ implementation of sparse_topn_source

	This function will return the same CSR matrix A it is given but with only the
	top ntop elements of each row sorted in descending order.

	Input:
		n_row: number of rows of A matrix
		n_col: number of columns of B matrix

		Ap, Aj, Ax: CSR expression of A matrix

		ntop: n top results

	Output by reference:
		Ap, Aj, Ax: CSR expression of A matrix
	N.B. A must be CSR format!!!
*/
template<typename T>
void sparse_hstack_source(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int Bp[],
		int Bj[],
		T Bx[]	//data of output
)
{
	std::vector<int> col_offsets(n_blocks + 1);
	col_offsets[0] = 0;
	std::partial_sum(n_col, n_col + n_blocks, col_offsets.data() + 1);
	int n_col_sum = col_offsets[n_blocks];

	std::vector<int*> bAp(n_blocks);
	std::vector<int*> bAj(n_blocks);
	std::vector<T*> bAx(n_blocks);
	bAp[0] = Ap;
	bAj[0] = Aj;
	bAx[0] = Ax;
	for (int b = 1; b < n_blocks; b++){
		bAp[b] = bAp[b - 1] + (n_row + 1);
		bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
		bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
	}
	Bp[0] = 0;
	int s = 0;
	for(int i = 0; i < n_row; i++){
		for (int b = 0; b < n_blocks; b++){
			int jj_start = bAp[b][i];
			int jj_end = bAp[b][i + 1];
			int offset = col_offsets[b];
			std::transform(&bAj[b][jj_start], &bAj[b][jj_end], &Bj[s], [&](int x){return (x + offset);});
			std::copy(&bAx[b][jj_start], &bAx[b][jj_end], &Bx[s]);
			s += jj_end - jj_start;
		}
		Bp[i + 1] = s;
	}
}
template void sparse_hstack_source<float>(int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], float Ax[], int Bp[], int Bj[], float Bx[]);
template void sparse_hstack_source<double>(int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], double Ax[], int Bp[], int Bj[], double Bx[]);
