import pandas as pd
import numpy as np
import pickle
import scipy.sparse
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from .ngrams import ngrams
from .database import get_connection

def prepare_data():
    """
    Prepare hyperparameters (vectorizer and sparse matrix) for string matcher.
    We load aliases from RDS merchant table, to train and transform vectorizer.
    """

    try:
        # Connect to the database server
        print('Connecting to the database...')
        engine = get_connection()
        print("Connection to database created successfully.")
    except Exception as err:
        print("Database connection could not be made due to the following error:")
        print(err)
        raise err

    with engine.connect() as conn:
        # Select all merchant aliases
        unique_aliases = pd.read_sql_query(
            sql="SELECT DISTINCT merchant_alias FROM merchants.merchants WHERE source_type != 'ABR'",
            con=conn.connection
        )
        unique_aliases.dropna(inplace=True)

        # Select all locations
        locations = pd.read_sql_query(
            sql="SELECT DISTINCT location FROM merchants.locations_cleaned",
            con=conn.connection
        )
        locations.dropna(inplace=True)

    # Create directory to store files
    files_dir = './files'
    os.makedirs(files_dir, exist_ok=True)

    # Save dataframes into CSV files
    unique_aliases.to_csv(f"{files_dir}/RDS_merchants.csv", index=False)
    master_merchants = unique_aliases["merchant_alias"]

    locations.to_csv(f"{files_dir}/RDS_locations.csv", index=False)
    master_locations = locations["location"]

    # Create and fit merchants Vectorizer
    vectorizer_merchants = TfidfVectorizer(min_df=1, analyzer=ngrams, dtype=np.float32)
    vectorizer_merchants.fit(master_merchants)

    vectorizer_locations = TfidfVectorizer(min_df=1, analyzer=ngrams, dtype=np.float32)
    vectorizer_locations.fit(master_locations)

    # Save merchants vectorizer into file
    with open(f"{files_dir}/vectorizer_merchants.pk", 'wb') as f_write:
        pickle.dump(vectorizer_merchants, f_write)

    # Save locations vectorizer into file
    with open(f"{files_dir}/vectorizer_locations.pk", 'wb') as f_write:
        pickle.dump(vectorizer_locations, f_write)

    # Generate Sparse TFIDF matrices from Master corpus using Vectorizer
    tf_idf_master_merchants = vectorizer_merchants.transform(master_merchants)
    tf_idf_master_locations = vectorizer_locations.transform(master_locations)

    # Save Sparse TFIDF matrix into file
    scipy.sparse.save_npz(f"{files_dir}/merchants_sparse.npz", tf_idf_master_merchants)
    scipy.sparse.save_npz(f"{files_dir}/locations_sparse.npz", tf_idf_master_locations)

    print("List of files in 'files' directory:", os.listdir(files_dir))

prepare_data()
