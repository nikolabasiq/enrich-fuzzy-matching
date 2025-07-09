#ifndef TOPN_PARALLEL_CPPCLASS_H
#define TOPN_PARALLEL_CPPCLASS_H

template<typename T>
struct rcd {
	int r;
	int c;
	T d;

	bool operator<(const rcd& a) const
    {
        return (a.d < d && a.r == r) || a.r > r;
    }

};


template<typename T>
extern int topn_parallel(
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_rows,
		int n_jobs
);

template<typename T>
extern void sparse_topn_parallel(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int ntop,
		int Bp[],
		int Bj[],
		T Bx[],	//data of output
		int n_jobs
);

template<typename T>
extern void sparse_hstack_parallel(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int Bp[],
		int Bj[],
		T Bx[],	//data of output
		int n_jobs
);

#endif //TOPN_PARALLEL_CPPCLASS_H
