import pandas as pd
import scipy.sparse
import pickle
import time
import os

from .string_matcher import fuzzy_match

# Load merchant data
merchant_master = pd.read_csv("data/RDS_merchants.csv")
merchant_master = merchant_master["merchant_alias"]

with open("data/vectorizer_merchants.pk", "rb") as f_read:
    merchant_vectorizer = pickle.load(f_read)

merchant_tf_idf_master = scipy.sparse.load_npz("data/merchants_sparse.npz")

# Load locations data
locations_master = pd.read_csv("data/RDS_locations.csv")
locations_master = locations_master["location"]

with open("data/vectorizer_locations.pk", "rb") as f_read:
    locations_vectorizer = pickle.load(f_read)

locations_tf_idf_master = scipy.sparse.load_npz("data/locations_sparse.npz")

def match_data(event, context):
    """
        Generate a Data Frame with matching results (aliases, similarity scores).

            Parameters:
                event (dict): A JSON-formatted document that contains data for a Lambda function to process.
                context: A context object, passed to the function at runtime, provides methods and properties.

            Returns:
                results (dict): Returns a Data Frame, which is converted to JSON.
    """
    print("Event:", event)
    original = pd.Series(event["value"])
    if event["type"] == "merchant":
        print("Entered if block - merchant:")
        master = merchant_master
        vectorizer = merchant_vectorizer
        tf_idf_master = merchant_tf_idf_master

    elif event["type"] == "location":
        print("Entered elif block - locations:")
        master = locations_master
        vectorizer = locations_vectorizer
        tf_idf_master = locations_tf_idf_master
    else:
        raise Exception(f"Matching not supported for type '{event['type']}'")

    start = time.time()
    print("Original:", original)

    minSimilarity = float(os.environ.get("MIN_SIMILARITY_PARAMETER"))
    print("minSimilarity:", minSimilarity)

    match = fuzzy_match(master=master,
                        original=original,
                        vectorizer=vectorizer,
                        tf_idf_master=tf_idf_master,
                        max_n_matches=5,
                        min_similarity=minSimilarity,
                        ignore_index=True)

    print("Match:", match)

    end = time.time()
    print("Time taken to fuzzy match: ", round((end - start), 2), "sec")

    # match.to_csv("data/results_1k.csv", index=False)

    results = match.to_json(orient="records")
    print("Results:", results)
    return results
