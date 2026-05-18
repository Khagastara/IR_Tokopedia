import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess, preprocess_dataframe

def build_index(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["document_clean"])
    return vectorizer, tfidf_matrix

def search(query, df, vectorizer, tfidf_matrix, top_k=10):
    query_clean = preprocess(query)

    if not query_clean.strip():
        return pd.DataFrame(), 0.0, 0.0

    # Cosine similarity scores
    query_vec = vectorizer.transform([query_clean])
    cosine_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # TF-IDF score per dokumen = rata-rata bobot kata query di dokumen tersebut
    query_terms = query_clean.split()
    feature_names = vectorizer.get_feature_names_out()
    vocab = {term: idx for idx, term in enumerate(feature_names)}

    tfidf_scores = np.zeros(tfidf_matrix.shape[0])
    matched_terms = [t for t in query_terms if t in vocab]

    if matched_terms:
        term_indices = [vocab[t] for t in matched_terms]
        tfidf_scores = np.array(
            tfidf_matrix[:, term_indices].mean(axis=1)
        ).flatten()

    # Ambil top_k berdasarkan cosine
    top_indices = cosine_scores.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["cosine_score"] = cosine_scores[top_indices]
    results["tfidf_score"]  = tfidf_scores[top_indices]

    # Filter skor 0
    results = results[results["cosine_score"] > 0].reset_index(drop=True)

    precision, recall = evaluate(results, cosine_scores, top_k)

    return results, precision, recall

def evaluate(results, scores, top_k):
    total_relevan_retrieved = len(results)
    total_retrieved         = len(results)
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