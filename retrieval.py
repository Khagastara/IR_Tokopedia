import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocessing import preprocess, preprocess_dataframe

def build_index(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["document_clean"])
    return vectorizer, tfidf_matrix

def search(query, df, vectorizer, tfidf_matrix, top_k=10):
    # Preprocess query
    query_clean = preprocess(query)

    if not query_clean.strip():
        return pd.DataFrame(), 0.0, 0.0

    # Hitung cosine similarity
    query_vec = vectorizer.transform([query_clean])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ambil top_k hasil
    top_indices = scores.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results["score"] = scores[top_indices]

    # Filter skor 0
    results = results[results["score"] > 0].reset_index(drop=True)

    # Hitung precision dan recall
    precision, recall = evaluate(results, scores, top_k)

    return results, precision, recall

def evaluate(results, scores, top_k):
    # Dokumen relevan = skor cosine > 0
    total_relevan_retrieved = len(results)
    total_retrieved = top_k
    total_relevan_tersedia = (scores > 0).sum()

    precision = (
        total_relevan_retrieved / total_retrieved
        if total_retrieved > 0 else 0.0
    )

    recall = (
        total_relevan_retrieved / total_relevan_tersedia
        if total_relevan_tersedia > 0 else 0.0
    )

    return round(precision, 4), round(recall, 4)