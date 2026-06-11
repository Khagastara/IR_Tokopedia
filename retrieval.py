import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess, preprocess_dataframe

def build_index(df):
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b"
    )
    tfidf_matrix = vectorizer.fit_transform(df["document_clean"])
    return vectorizer, tfidf_matrix

def get_query_term_weights(query_clean, vectorizer):
    query_vec = vectorizer.transform([query_clean])
    feature_names = vectorizer.get_feature_names_out()
    scores = query_vec.toarray().flatten()
    top_idx = scores.argsort()[::-1]
    terms = [
        {"term": feature_names[i], "score": round(float(scores[i]), 4)}
        for i in top_idx if scores[i] > 0
    ]
    return terms

def search(query, df, vectorizer, tfidf_matrix, top_k=10):
    query_clean = preprocess(query)

    if not query_clean.strip():
        return pd.DataFrame(), 0.0, 0.0, []

    query_vec = vectorizer.transform([query_clean])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = cosine_scores.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["cosine_score"] = cosine_scores[top_indices]
    results = results[results["cosine_score"] > 0].reset_index(drop=True)

    term_weights = get_query_term_weights(query_clean, vectorizer)
    precision, recall = evaluate(results, cosine_scores, top_k)

    return results, precision, recall, term_weights

def evaluate(results, scores, top_k):
    total_relevan_retrieved = len(results)
    total_retrieved         = top_k
    total_relevan_tersedia  = (scores > 0).sum()

    precision = (
        total_relevan_retrieved / total_retrieved
        if total_retrieved > 0 else 0.0
    )
    recall = (
        total_relevan_retrieved / total_relevan_tersedia
        if total_relevan_tersedia > 0 else 0.0
    )

    return round(precision, 4), round(recall, 4)