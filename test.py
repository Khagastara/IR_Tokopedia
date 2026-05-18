import pandas as pd
from preprocessing import preprocess_dataframe
from retrieval import build_index, search

df = pd.read_csv("data/produk_tokopedia.csv")
df = preprocess_dataframe(df)
df.to_csv("data/produk_clean.csv", index=False)

vectorizer, tfidf_matrix = build_index(df)

query = "laptop asus"
results, precision, recall = search(query, df, vectorizer, tfidf_matrix, top_k=10)

print(f"Query     : {query}")
print(f"Precision : {precision}")
print(f"Recall    : {recall}")
print(results[["product_name", "score"]].head())