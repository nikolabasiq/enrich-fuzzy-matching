import unittest
import pandas as pd
import numpy as np
from scipy.sparse.csr import csr_matrix
from topn import awesome_topn, awesome_hstack_topn, awesome_hstack
from unittest.mock import patch

class topnTest(unittest.TestCase):
    def test_topn(self):
        (r, c, d) = (
            np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int64),
            np.array([0, 1, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 5, 3, 4, 3, 4, 3, 4, 3, 4, 4, 3, 5], dtype=np.int64),
            np.array([0.99999994, 0.25998905, 1.0000001 , 0.6412701 , 0.25998905, 1.        , 0.6412701 , 1.        , 0.6412701 , 1.        , 0.6412701 , 0.33135417, 0.21248752, 0.808251  , 0.808251  , 0.6412701 , 0.21248752, 1.        , 0.33135417, 1.        , 0.33135417, 1.        , 0.33135417, 1.0000001 , 0.33135417, 0.9999999 ], dtype=np.float32)
        )
        ntop = 6
        n_rows = 7
        n_cols = 6
        rcd = pd.DataFrame({'r': r, 'c': c, 'd': d})
        rcd_topn = rcd.set_index('c').groupby('r')['d'].nlargest(ntop).reset_index().sort_values(['r', 'd'], ascending = [True, False])
        x_r = rcd_topn['r'].to_numpy()
        x_c = rcd_topn['c'].to_numpy()
        x_d = rcd_topn['d'].to_numpy()
        x_M = csr_matrix((x_d, (x_r, x_c)), shape=(n_rows, n_cols))
        
        o_r, o_c, o_d = awesome_topn(r, c, d, ntop, n_jobs=7)
        
        np.testing.assert_array_equal(x_d, o_d)
        np.testing.assert_array_equal(x_c, o_c)
        np.testing.assert_array_equal(x_r, o_r)

        o_r, o_c, o_d = awesome_topn(r, c, d, ntop, n_rows=n_rows, n_jobs=7)
        M = csr_matrix((o_d, o_c, o_r), shape=(n_rows, n_cols))
        assert (x_M != M).nnz == 0

        o_r, o_c, o_d = awesome_topn(r, c, d, ntop, n_rows=n_rows, n_jobs=1)
        M1 = csr_matrix((o_d, o_c, o_r), shape=(n_rows, n_cols))
        assert (x_M != M1).nnz == 0
       
    def test_hstack_topn(self):
        (r, c, d) = (
            np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int64),
            np.array([0, 1, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 5, 3, 4, 3, 4, 3, 4, 3, 4, 4, 3, 5], dtype=np.int64),
            np.array([0.99999994, 0.25998905, 1.0000001 , 0.6412701 , 0.25998905, 1.        , 0.6412701 , 1.        , 0.6412701 , 1.        , 0.6412701 , 0.33135417, 0.21248752, 0.808251  , 0.808251  , 0.6412701 , 0.21248752, 1.        , 0.33135417, 1.        , 0.33135417, 1.        , 0.33135417, 1.0000001 , 0.33135417, 0.9999999 ], dtype=np.float32)
        )
        ntop = 4
        n_rows = 7
        n_cols = 6
        rcd = pd.DataFrame({'r': r, 'c': c, 'd': d})
        rcd_topn = rcd.set_index('c').groupby('r')['d'].nlargest(ntop).reset_index().sort_values(['r', 'd'], ascending = [True, False])
        x_r = rcd_topn['r'].to_numpy()
        x_c = rcd_topn['c'].to_numpy()
        x_d = rcd_topn['d'].to_numpy()
        x_M = csr_matrix((x_d, (x_r, x_c)), shape=(n_rows, n_cols))
        
        M0 = csr_matrix((d, (r, c)), shape=(n_rows, n_cols))
        
        M1 = awesome_hstack_topn([M0], ntop, use_threads=False)
        M2 = awesome_hstack_topn([M0[:, :2].tocsr(), M0[:, 2:].tocsr()], ntop, use_threads=False)

        assert (x_M != M1).nnz == 0
        assert (x_M != M2).nnz == 0
        
        M1 = awesome_hstack_topn([M0], ntop, use_threads=True, n_jobs=7)
        M2 = awesome_hstack_topn([M0[:, :2].tocsr(), M0[:, 2:].tocsr()], ntop, use_threads=True, n_jobs=7)

        assert (x_M != M1).nnz == 0
        assert (x_M != M2).nnz == 0
        
    def test_hstack(self):
        (r, c, d) = (
            np.array([0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6], dtype=np.int64),
            np.array([0, 1, 1, 2, 0, 2, 1, 2, 1, 2, 1, 2, 1, 0, 5, 3, 4, 3, 4, 3, 4, 3, 4, 4, 3, 5], dtype=np.int64),
            np.array([0.99999994, 0.25998905, 1.0000001 , 0.6412701 , 0.25998905, 1.        , 0.6412701 , 1.        , 0.6412701 , 1.        , 0.6412701 , 0.33135417, 0.21248752, 0.808251  , 0.808251  , 0.6412701 , 0.21248752, 1.        , 0.33135417, 1.        , 0.33135417, 1.        , 0.33135417, 1.0000001 , 0.33135417, 0.9999999 ], dtype=np.float32)
        )
        n_rows = 7
        n_cols = 6
        M0 = csr_matrix((d, (r, c)), shape=(n_rows, n_cols))
        
        M1 = awesome_hstack([M0[:, :2].tocsr(), M0[:, 2:].tocsr()], use_threads=False)
        assert (M0 != M1).nnz == 0
        
        M2 = awesome_hstack([M0[:, :2].tocsr(), M0[:, 2:].tocsr()], use_threads=True, n_jobs=3)
        assert (M0 != M2).nnz == 0
        
        M3 = awesome_hstack([M0[:, :2].tocsr(), M0[:, 2:].tocsr()], use_threads=True, n_jobs=7)
        assert (M0 != M3).nnz == 0
       
        
if __name__ == '__main__':
    unittest.main()
