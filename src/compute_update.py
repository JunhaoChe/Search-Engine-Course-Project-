import math
import os
import json
import subprocess
import sys
from collections import defaultdict
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_idf(total_documents, index_directory="indexes_by_partition"):
    """
    Calculate IDF values for each token across all partitions.
    """
    token_doc_count = defaultdict(int)  # Count how many documents each token appears in

    # Iterate through index files to gather document frequency (DF)
    for filename in os.listdir(index_directory):
        if filename.startswith("index_") and filename.endswith(".json"):
            file_path = os.path.join(index_directory, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                    for token, postings in index_data.items():
                        token_doc_count[token] += len(postings)  # DF = number of documents a term appears in
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    # Compute IDF for each token
    idf_values = {}
    for token, doc_count in token_doc_count.items():
        idf_values[token] = math.log((total_documents + 1) / (doc_count + 1)) + 1  # Smoothing added
    return idf_values

def update_tf_idf_in_index(idf_values, index_directory="indexes_by_partition"):
    """
    Update each partition with the computed TF-IDF values.
    """
    for filename in os.listdir(index_directory):
        if filename.startswith("index_") and filename.endswith(".json"):
            file_path = os.path.join(index_directory, filename)

            try:
                # Load the existing index
                with open(file_path, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)

                # Update TF-IDF values
                for token, postings in index_data.items():
                    if token in idf_values:
                        idf = idf_values[token]
                        for posting in postings:
                            for doc_id, attributes in posting.items():
                                tf = attributes["f"]  # Retrieve normalized TF
                                attributes["s"] = tf * idf  # Calculate and add TF-IDF

                # Write updated index back to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(index_data, f, indent=4)

            except Exception as e:
                print(f"Error updating {file_path}: {e}")

def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:  # Avoid division by zero
        return vector
    return vector / norm


def my_cosine_similarity(query_vector, doc_vector):
    query_vector = normalize(query_vector)
    doc_vector = normalize(doc_vector)

    # Calculate cosine similarity after normalization
    dot_product = np.dot(query_vector, doc_vector)
    return dot_product


def get_common_doc_s_values(lists):
    """
    Find the document and its sum of tf_idf values

    :param lists: Multiple lists of (id, tf_idf)
    :return: Dictionary {doc_id: tf_idf} for common doc_ids
    """
    if not lists or len(lists) == 0:
        return {}
    sorted_lists = sorted(lists, key=lambda _list: len(_list))
    id_tf_idf = defaultdict()
    for l in sorted_lists:
        for item in l:
            id = item[0]
            tf_idf = item[1]
            id_tf_idf[id] += tf_idf

    return id_tf_idf

def rank_documents(document_list):
    """
        Rank documents based on their average cosine similarity score.

        Args:
            document_list (list of tuples): List of (id, content) tuples.

        Returns:
            list: List of document IDs sorted by similarity score in descending order.
    """
    if not document_list:
        return []

    # Extract document IDs and contents
    doc_ids, contents = zip(*document_list)

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1, 2))
    tf_idf_matrix = vectorizer.fit_transform(contents)

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tf_idf_matrix)

    # Compute average similarity score per document
    avg_similarities = np.mean(cosine_sim, axis=1)

    # Create dictionary of {doc_id: similarity_score}
    doc_scores = {doc_id: score for doc_id, score in zip(doc_ids, avg_similarities)}

    # Sort by similarity score in descending order and return the list of doc IDs
    sorted_doc_ids = [int(doc_id) for doc_id, _ in sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)]

    return sorted_doc_ids

def compare_with_query(document, query):
    """
    Compute the cosine similarity between a document and a query.

    :param document: A string representing the document content.
    :param query: A string representing the search query.
    :return: Cosine similarity score between the document and the query.
    """
    if not document or not query:
        return 0.0  # Return 0 similarity if either input is empty

    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), sublinear_tf=True)

    # Fit and transform both the document and query
    tfidf_matrix = vectorizer.fit_transform([document, query])

    # Compute cosine similarity
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

    return similarity_score




