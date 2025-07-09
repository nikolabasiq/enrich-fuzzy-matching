#ifndef TOPN_SOURCE_CPPCLASS_H
#define TOPN_SOURCE_CPPCLASS_H

template<typename T>
struct Candidate {
	int index;
	T value;

	bool operator<(const Candidate& a) const
    {
        return a.value < value;
    }

};

template<typename T>
extern void sparse_topn_source(
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
);


template<typename T>
extern void sparse_hstack_source(
		int n_blocks,
		int n_row,
		int n_col[],
		int Ap[],
		int Aj[],
		T Ax[],	//data of A
		int Bp[],
		int Bj[],
		T Bx[]	//data of output
);

#endif //TOPN_SOURCE_CPPCLASS_H
