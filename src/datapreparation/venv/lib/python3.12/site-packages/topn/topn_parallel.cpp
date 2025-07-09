#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>

#include "./topn_source.h"
#include "./topn_parallel.h"

using namespace std;

struct job_range_type {int begin; int end;};

void distribute_load(
		int load_sz,
		int n_jobs,
		vector<job_range_type> &ranges
)
{
	// share the load among jobs:
	int equal_job_load_sz = load_sz/n_jobs;
	int rem = load_sz % n_jobs;
	ranges.resize(n_jobs);

	int start = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		ranges[job_nr].begin = start;
		ranges[job_nr].end = start + equal_job_load_sz + ((job_nr < rem)? 1 : 0);
		start = ranges[job_nr].end;
	}
}

template<typename T>
void inner_gather(
		int job_sz,
		rcd<T> v_rcd[],
		int r[],
		int c[],
		T d[]
)
{
	for (int i = 0; i < job_sz; i++){
		r[i] = v_rcd[i].r;
		c[i] = v_rcd[i].c;
		d[i] = v_rcd[i].d;
	}
}
template void inner_gather<float>(int job_sz, rcd<float> v_rcd[], int r[], int c[], float d[]);
template void inner_gather<double>(int job_sz, rcd<double> v_rcd[], int r[], int c[], double d[]);


template<typename T>
void inner_gather_csr(
		int job_sz,
		rcd<T> v_rcd[],
		int r[],
		int c[],
		T d[]
)
{
	for (int i = 0; i < job_sz; i++){
		r[v_rcd[i].r]++;
		c[i] = v_rcd[i].c;
		d[i] = v_rcd[i].d;
	}
}
template void inner_gather_csr<float>(int job_sz, rcd<float> v_rcd[], int r[], int c[], float d[]);
template void inner_gather_csr<double>(int job_sz, rcd<double> v_rcd[], int r[], int c[], double d[]);


template<typename T>
void inner_copy_to_v_rcd(
		job_range_type job_range,
		rcd<T> dest[],
		int r[],
		int c[],
		T d[]
)
{
	rcd<T>* dest_cursor = &dest[0];
	int* r_cursor = &r[0];
	int* c_cursor = &c[0];
	T* d_cursor = &d[0];

	for (int i = job_range.begin; i < job_range.end; i++){
		dest_cursor->r = *(r++);
		dest_cursor->c = *(c++);
		(dest_cursor++)->d = *(d++);
	}
}
template void inner_copy_to_v_rcd<float>(job_range_type job_range, rcd<float> dest[], int r[], int c[], float d[]);
template void inner_copy_to_v_rcd<double>(job_range_type job_range, rcd<double> dest[], int r[], int c[], double d[]);

template<typename T>
void inner_pick_nlargest(
		job_range_type job_range,
		rcd<T> v_rcd[],
		int ntop,
		int* new_end
)
{
	rcd<T>* write_cusor = &v_rcd[0];
	rcd<T>* read_cusor = &v_rcd[0];
	int c = 0;
	int r, prev_r = v_rcd[0].r;

	for (int i = job_range.begin; i < job_range.end; i++){

		r = read_cusor->r;
		if (prev_r != r) c = 0;
		if (c < ntop) *(write_cusor++) = *read_cusor;
		read_cusor++;
		prev_r = r;
		c++;
	}
	*new_end = write_cusor - &v_rcd[0];
}

template void inner_pick_nlargest<float>(job_range_type job_range, rcd<float> v_rcd[], int ntop, int* new_end);
template void inner_pick_nlargest<double>(job_range_type job_range, rcd<double> v_rcd[], int ntop, int* new_end);


/*
 * The idea of the function "find_next_boundary" is how to get from ^ to *
 * (i.e., from inside a 1D-zone to just outside it in the next zone)
 *
 * previous zone     current zone         next zone
 * _______________|_________________|__________________
 *                       ^           *
 */
template<typename T>
int find_next_boundary(rcd<T> v_rcd[], int x, int end)
{
	if (v_rcd[x - 1].r == v_rcd[end - 1].r) return end;
	while (v_rcd[x - 1].r == v_rcd[x].r){ // non-boundary condition
		int mid = (x + end)/2;
		if (mid == x) return end;
		else if (v_rcd[x].r == v_rcd[mid - 1].r) x = mid;
		else end = mid;
	}
	return x;
}

template int find_next_boundary<float>(rcd<float> v_rcd[], int x, int end);
template int find_next_boundary<double>(rcd<double> v_rcd[], int x, int end);

/*
 * The idea of the function "find_prev_boundary" is how to get from ^ to *
 * (i.e., from inside a 1D-zone to its start just outside the previous zone)
 *
 * previous zone     current zone         next zone
 * _______________|_________________|__________________
 *                 *        ^
 */
template<typename T>
int find_prev_boundary(rcd<T> v_rcd[], int begin, int x)
{
	if (v_rcd[begin].r == v_rcd[x].r) return begin;
	while (v_rcd[x - 1].r == v_rcd[x].r){ // non-boundary condition
		int mid = (begin + x)/2;
		if (v_rcd[x].r == v_rcd[mid].r) x = mid;
		else begin = mid;
	}
	return x;
}

template int find_prev_boundary<float>(rcd<float> v_rcd[], int x, int end);
template int find_prev_boundary<double>(rcd<double> v_rcd[], int x, int end);


/*
	r, c, and d are 1D arrays all of the same length n.
	This function will output arrays rn, cn, and dn of length N <= n such
    that the set of triples {(rn[i], cn[i], dn[i]) : 0 < i < N} is a subset of
    {(r[j], c[j], d[j]) : 0 < j < n} and that for every distinct value
    x = rn[i], dn[i] is among the first ntop existing largest d[j]'s whose
    r[j] = x.

    Input:
        r and c: two 1D integer arrays of the same length
        d: 1D array of single or double precision floating point type of the
        	same length as r or c
        ntop maximum number of
        use_threads: use multi-thread or not
        n_jobs: number of threads, must be >= 1

    Output:
        rn, cn, dn where rn, cn, dn are all arrays as described above and will
        	replace r, c, d.
        returns the length of the output arrays
*/
template<typename T>
int topn_parallel(
		int n,
		int r[],
		int c[],
		T d[],
		int ntop,
		int n_rows,
		int n_jobs
)
{
	// distribute the load among jobs/threads
	vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n, n_jobs, job_ranges);

	// initialize temporary storage and copy in parallel
	vector<rcd<T>> v_rcd(n);
	vector<thread> thread_list(n_jobs);
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_copy_to_v_rcd<T>,
				job_ranges[job_nr],
				&v_rcd[job_ranges[job_nr].begin],
				&r[job_ranges[job_nr].begin],
				&c[job_ranges[job_nr].begin],
				&d[job_ranges[job_nr].begin]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	// sort the data in temporary location
	sort(&v_rcd[0], &v_rcd[n]);

	// adjust the job-range boundaries to coincide with row boundaries:
	for (int job_nr = 1; job_nr < n_jobs; job_nr++) {
		// This condition is always enforced:
		job_ranges[job_nr].begin = job_ranges[job_nr - 1].end;

		int x = job_ranges[job_nr].begin;	// x marks the spot
		int f = job_ranges[job_nr - 1].begin;	// f: floor
		int c = n;	// c: ceiling
		int x0 = find_prev_boundary<T>(v_rcd.data(), f, x);
		int x1 = find_next_boundary<T>(v_rcd.data(), x, c);
		if (f < x0 && (x0 - x) >= (x - x1)){
			job_ranges[job_nr - 1].end = x0;
			job_ranges[job_nr].begin = x0;
		}
		else {
			job_ranges[job_nr - 1].end = x1;
			job_ranges[job_nr].begin = x1;
		}
		if (job_ranges[job_nr].begin > job_ranges[job_nr].end)
			job_ranges[job_nr].end = job_ranges[job_nr].begin;
	}

	// pick n largest elements
	// initialize aggregate:
	vector<int> sub_counts(n_jobs, 0);

	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {
		thread_list[job_nr] = thread(
				inner_pick_nlargest<T>,
				job_ranges[job_nr],
				&v_rcd[job_ranges[job_nr].begin],
				ntop,
				&sub_counts[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	vector<int> dest_job_starts(n_jobs + 1);
	dest_job_starts[0] = 0;
	partial_sum(sub_counts.begin(), sub_counts.end(), dest_job_starts.begin() + 1);

	// gather the results:
	if (n_rows < 0){
		for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

			thread_list[job_nr] = thread(
					inner_gather<T>,
					sub_counts[job_nr],
					&v_rcd[job_ranges[job_nr].begin],
					&r[dest_job_starts[job_nr]],
					&c[dest_job_starts[job_nr]],
					&d[dest_job_starts[job_nr]]
			);
		}
		for (int job_nr = 0; job_nr < n_jobs; job_nr++)
			thread_list[job_nr].join();
	}
	else {
		vector<int> indptr(n_rows, 0);
		for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

			thread_list[job_nr] = thread(
					inner_gather_csr<T>,
					sub_counts[job_nr],
					&v_rcd[job_ranges[job_nr].begin],
					&indptr[0],
					&c[dest_job_starts[job_nr]],
					&d[dest_job_starts[job_nr]]
			);
		}
		for (int job_nr = 0; job_nr < n_jobs; job_nr++)
			thread_list[job_nr].join();

		r[0] = 0;
		partial_sum(indptr.data(), indptr.data() + n_rows, r + 1);
	}
	return dest_job_starts[n_jobs];
}
template int topn_parallel<float>(int n, int r[], int c[], float d[], int ntop, int n_rows, int n_jobs);
template int topn_parallel<double>(int n, int r[], int c[], double d[], int ntop, int n_rows, int n_jobs);

template<typename T>
void inner_sparse_topn(
		job_range_type job_range,
		int n_blocks,
		int col_offset[],
		int ntop_inner,
		int* bAp[],
		int* bAj[],
		T* bAx[],
		vector<Candidate<T>>* real_candidates,
		vector<int>* row_nnz,
		int* total
)
{
	// calculate maximum memory space needed to be reserved for sorted values
	int mem_sz = 0;
	for (int b = 0; b < n_blocks; b++) mem_sz += bAp[b][job_range.end] - bAp[b][job_range.begin];
	real_candidates->reserve(mem_sz);

	row_nnz->resize(job_range.end - job_range.begin);
	int* row_nnz_ptr = row_nnz->data();

	for(int i = job_range.begin; i < job_range.end; i++){

		size_t sz = real_candidates->size();
		for (int b = 0; b < n_blocks; b++){
			int jj_start = bAp[b][i];
			int jj_end   = bAp[b][i+1];

			for(int jj = jj_start; jj < jj_end; jj++){
				Candidate<T> c;
				c.index = bAj[b][jj] + col_offset[b];
				c.value = bAx[b][jj];
				real_candidates->push_back(c);
			}
		}
		int len = (int) (real_candidates->size() - sz);
		Candidate<T>* candidate_arr_begin = real_candidates->data() + sz;
		if (len > ntop_inner){
			partial_sort(
					candidate_arr_begin,
					candidate_arr_begin + ntop_inner,
					candidate_arr_begin + len
			);
			len = ntop_inner;
		}
		else {
			sort(
					candidate_arr_begin,
					candidate_arr_begin + len
			);
		}

		real_candidates->resize(sz + (size_t) len);
		*(row_nnz_ptr++) = len;
		(*total) += len;
	}
}
template void inner_sparse_topn<float>( job_range_type job_range, int n_blocks, int col_offset[], int ntop_inner, int* bAp[], int* bAj[], float* bAx[], vector<Candidate<float>>* real_candidates, vector<int>* row_nnz, int* total);
template void inner_sparse_topn<double>( job_range_type job_range, int n_blocks, int col_offset[], int ntop_inner, int* bAp[], int* bAj[], double* bAx[], vector<Candidate<double>>* real_candidates, vector<int>* row_nnz, int* total);


template<typename T>
void inner_get_nnz_hstack(
		job_range_type job_range,
		int n_blocks,
		int* bAp[],
		int* bAj[],
		T* bAx[],
		int* total
)
{
	// calculate maximum memory space needed to be reserved for sorted values
	int mem_sz = 0;
	for (int b = 0; b < n_blocks; b++) mem_sz += bAp[b][job_range.end] - bAp[b][job_range.begin];
	(*total) = mem_sz;
}
template void inner_get_nnz_hstack<float>( job_range_type job_range, int n_blocks, int* bAp[], int* bAj[], float* bAx[], int* total);
template void inner_get_nnz_hstack<double>( job_range_type job_range, int n_blocks, int* bAp[], int* bAj[], double* bAx[], int* total);


template<typename T>
void inner_sparse_hstack(
		job_range_type job_range,
		int n_blocks,
		int col_offset[],
		int* bAp[],
		int* bAj[],
		T* bAx[],
		int Bp[],
		int Bj[],
		T Bx[],
		int start
)
{
	int s = start;
	for(int i = job_range.begin; i < job_range.end; i++){

		for (int b = 0; b < n_blocks; b++){
			int jj_start = bAp[b][i];
			int jj_end   = bAp[b][i+1];
			int offset = col_offset[b];

			transform(&bAj[b][jj_start], &bAj[b][jj_end], &Bj[s], [&](int x){return (x + offset);});
			copy(&bAx[b][jj_start], &bAx[b][jj_end], &Bx[s]);
			s += jj_end - jj_start;
		}
		Bp[i + 1] = s;
	}
}
template void inner_sparse_hstack<float>( job_range_type job_range, int n_blocks, int col_offset[], int* bAp[], int* bAj[], float* bAx[], int Bp[], int Bj[], float Bx[], int start);
template void inner_sparse_hstack<double>( job_range_type job_range, int n_blocks, int col_offset[], int* bAp[], int* bAj[], double* bAx[], int Bp[], int Bj[], double Bx[], int start);


template<typename T>
void inner_gather_v1(
		job_range_type job_range,
		int Cp[],
		int Cp_start,
		int vCj_start[],
		T vCx_start[],
		vector<Candidate<T>>* real_candidates,
		vector<int>* row_nnz
)
{
	Candidate<T>* c = real_candidates->data();
	int* vCj_cursor = &vCj_start[Cp_start];
	T* vCx_cursor = &vCx_start[Cp_start];

	int Cp_i = Cp_start;
	int* row_nnz_ptr = row_nnz->data();

	for (int i = job_range.begin; i < job_range.end; i++){
		for (int j = 0; j < (*row_nnz_ptr); j++){
			*(vCj_cursor++) = c->index;
			*(vCx_cursor++) = (c++)->value;
		}
		Cp_i += *(row_nnz_ptr++);
		Cp[i + 1] = Cp_i;
	}
}
template void inner_gather_v1<float>(job_range_type job_range, int Cp[], int Cp_start, int vCj_start[], float vCx_start[], vector<Candidate<float>>* real_candidates, vector<int>* row_nnz);
template void inner_gather_v1<double>(job_range_type job_range, int Cp[], int Cp_start, int vCj_start[], double vCx_start[], vector<Candidate<double>>* real_candidates, vector<int>* row_nnz);


template<typename T>
void sparse_topn_parallel(
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
)
{
	vector<int> col_offsets(n_blocks + 1);
	col_offsets[0] = 0;
	partial_sum(n_col, n_col + n_blocks, col_offsets.data() + 1);
	int n_col_sum = col_offsets[n_blocks];

	vector<int*> bAp(n_blocks);
	vector<int*> bAj(n_blocks);
	vector<T*> bAx(n_blocks);
	bAp[0] = Ap;
	bAj[0] = Aj;
	bAx[0] = Ax;
	for (int b = 1; b < n_blocks; b++){
		bAp[b] = bAp[b - 1] + (n_row + 1);
		bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
		bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
	}

	vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	vector<vector<Candidate<T>>> real_candidates(n_jobs);
	vector<vector<int>> row_nnz(n_jobs);

	// initialize aggregates:
	vector<int> sub_total(n_jobs, 0);

	vector<thread> thread_list(n_jobs);

	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_sparse_topn<T>,
				job_ranges[job_nr],
				n_blocks,
				&col_offsets[0], ntop,
				&bAp[0], &bAj[0], &bAx[0],
				&real_candidates[job_nr],
				&row_nnz[job_nr],
				&sub_total[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	vector<int> nnz_job_starts(n_jobs + 1);
	nnz_job_starts[0] = 0;
	partial_sum(sub_total.begin(), sub_total.end(), nnz_job_starts.begin() + 1);

	Bp[0] = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_gather_v1<T>,
				job_ranges[job_nr],
				Bp,
				nnz_job_starts[job_nr],
				Bj,
				Bx,
				&real_candidates[job_nr],
				&row_nnz[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();
}
template void sparse_topn_parallel<float>( int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], float Ax[], int ntop, int Bp[], int Bj[], float Bx[], int n_jobs);
template void sparse_topn_parallel<double>( int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], double Ax[], int ntop, int Bp[], int Bj[], double Bx[], int n_jobs);

template<typename T>
void inner_gather_v3(
		job_range_type job_range,
		int Cp[],
		int Cp_start,
		int Cj[],
		T Cx[],
		vector<int>* real_indices,
		vector<T>* real_values,
		vector<int>* row_nnz
)
{
	if (job_range.begin >= job_range.end) return;

	int* nnz_begin = row_nnz->data();
	int* nnz_end = nnz_begin + row_nnz->size();

	int* Cp_begin = &Cp[job_range.begin + 1];

	(*nnz_begin) += Cp_start;
	partial_sum(nnz_begin, nnz_end, Cp_begin);

	int* tcj_begin = real_indices->data();
	int* tcj_end = tcj_begin + real_indices->size();

	T* tcx_begin = real_values->data();
	T* tcx_end = tcx_begin + real_values->size();

	int* Cj_begin = &Cj[Cp_start];
	T* Cx_begin = &Cx[Cp_start];

	copy(tcj_begin, tcj_end, Cj_begin);
	copy(tcx_begin, tcx_end, Cx_begin);
}
template void inner_gather_v3<float>(job_range_type job_range, int Cp[], int Cp_start, int Cj[], float Cx[], vector<int>* real_indices, vector<float>* real_values, vector<int>* row_nnz);
template void inner_gather_v3<double>(job_range_type job_range, int Cp[], int Cp_start, int Cj[], double Cx[], vector<int>* real_indices, vector<double>* real_values, vector<int>* row_nnz);


template<typename T>
void sparse_hstack_parallel(
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
)
{
	vector<int> col_offsets(n_blocks + 1);
	col_offsets[0] = 0;
	partial_sum(n_col, n_col + n_blocks, col_offsets.data() + 1);
	int n_col_sum = col_offsets[n_blocks];

	vector<int*> bAp(n_blocks);
	vector<int*> bAj(n_blocks);
	vector<T*> bAx(n_blocks);
	bAp[0] = Ap;
	bAj[0] = Aj;
	bAx[0] = Ax;
	for (int b = 1; b < n_blocks; b++){
		bAp[b] = bAp[b - 1] + (n_row + 1);
		bAj[b] = bAj[b - 1] + bAp[b - 1][n_row];
		bAx[b] = bAx[b - 1] + bAp[b - 1][n_row];
	}

	vector<job_range_type> job_ranges(n_jobs);
	distribute_load(n_row, n_jobs, job_ranges);

	// initialize aggregates:
	vector<int> sub_total(n_jobs, 0);

	vector<thread> thread_list(n_jobs);

	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_get_nnz_hstack<T>,
				job_ranges[job_nr],
				n_blocks,
				&bAp[0], &bAj[0], &bAx[0],
				&sub_total[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();

	vector<int> nnz_job_starts(n_jobs);
	nnz_job_starts[0] = 0;
	partial_sum(sub_total.data(), sub_total.data() + (n_jobs - 1), nnz_job_starts.data() + 1);

	Bp[0] = 0;
	for (int job_nr = 0; job_nr < n_jobs; job_nr++) {

		thread_list[job_nr] = thread(
				inner_sparse_hstack<T>,
				job_ranges[job_nr],
				n_blocks,
				&col_offsets[0],
				&bAp[0], &bAj[0], &bAx[0],
				Bp, Bj, Bx,
				nnz_job_starts[job_nr]
		);
	}

	for (int job_nr = 0; job_nr < n_jobs; job_nr++)
		thread_list[job_nr].join();
}
template void sparse_hstack_parallel<float>( int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], float Ax[], int Bp[], int Bj[], float Bx[], int n_jobs);
template void sparse_hstack_parallel<double>( int n_blocks, int n_row, int n_col[], int Ap[], int Aj[], double Ax[], int Bp[], int Bj[], double Bx[], int n_jobs);
