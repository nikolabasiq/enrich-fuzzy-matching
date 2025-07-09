import pandas as pd
import numpy as np
import multiprocessing

from scipy.sparse.csr import csr_matrix
from typing import Union
from sparse_dot_topn_for_blocks import awesome_cossim_topn


def build_matches(tf_idf_original: csr_matrix,
                  tf_idf_master: csr_matrix,
                  max_n_matches: int,
                  min_similarity: float):
    """Builds the cosine similarity matrix of two csr matrices"""

    tf_idf_master = tf_idf_master.transpose()

    nnz_rows = np.full(tf_idf_original.shape[0], 0, dtype=np.int32)
    number_of_processes: int = multiprocessing.cpu_count() - 1
    optional_kwargs = {
        'return_best_ntop': True,
        'sort': True,
        'use_threads': number_of_processes > 1,
        'n_jobs': number_of_processes}

    return awesome_cossim_topn(tf_idf_original,
                               tf_idf_master,
                               max_n_matches,
                               nnz_rows,
                               min_similarity,
                               **optional_kwargs)


def get_matches_list(matches: csr_matrix):
    """Returns a list of all the indices of matches"""

    r, c = matches.nonzero()
    d = matches.data
    return pd.DataFrame({'master_side': c.astype(np.int64),
                         'original_side': r.astype(np.int64),
                         'similarity': d})


def get_both_sides(master,
                   original,
                   matches_list,
                   generic_name=('string', 'string'),
                   drop_index=False):
    """Extract names and indexes for master and original"""

    l_name, r_name = generic_name
    # left = master if master.name else master.rename(l_name)
    left = master.rename(l_name)
    left = left.iloc[matches_list.master_side].reset_index(drop=drop_index)
    # right = original if original.name else original.rename(r_name)
    right = original.rename(r_name)
    right = right.iloc[matches_list.original_side].reset_index(drop=drop_index)
    return left, (right if isinstance(right, pd.Series) else right[right.columns[::-1]])


def prefix_column_names(data: Union[pd.Series, pd.DataFrame], prefix: str):
    if isinstance(data, pd.DataFrame):
        return data.rename(columns={c: f"{prefix}{c}" for c in data.columns})
    else:
        return data.rename(f"{prefix}{data.name}")


def fuzzy_match(master,
                original,
                vectorizer,
                tf_idf_master,
                min_similarity=0.8,
                max_n_matches=100,
                ignore_index=False):
    """
    master (pd.Series): List of strings to be matched against
    original (pd.Series): List of strings to be matched
    vectorizer (klearn.feature_extraction.text.TfidfVectorizer): fitted Vectorizer on master
    tf_idf_master (csr_matrix): transformed vectorizer on master
    min_similarity (float): The minimum cosine similarity for two strings to be considered a match
    max_n_matches (int): The maximum number of matching strings in master allowed per string in original
    ignore_index (Bool): Determines whether indexes are ignored in final output or not
    """

    # Generate Sparse TFIDF matrix from original corpus using Vectorizer
    tf_idf_original = vectorizer.transform(original.iloc[slice(*(None, None))])

    # Calculate cosine similarity (between fitted vectorizer on master and on original)
    matches, _ = build_matches(tf_idf_original, tf_idf_master, max_n_matches, min_similarity)

    # Formate awesome_cossim_topn output(matches) into DataFrame
    matches_list = get_matches_list(matches)

    # Export master names (left_side), original names (right_side) and their similarities in relevant order
    left_side, right_side = get_both_sides(master, original, matches_list, drop_index=ignore_index)
    similarity = matches_list.similarity.reset_index(drop=True)

    left_prefix: str = 'matched_'  # used to set the left name of output column.
    right_prefix: str = 'input_'  # used to set the right name of output column.

    # Returns a DataFrame with all the matches and their cosine similarity.
    return pd.concat(
        [
            prefix_column_names(left_side, left_prefix),
            similarity,
            prefix_column_names(right_side, right_prefix)
        ],
        axis=1
    )
