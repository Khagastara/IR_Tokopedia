from flask import Flask, render_template, request
from retrieval import build_index, search
from preprocessing import preprocess_dataframe
import pandas as pd
import os

app = Flask(__name__)

try:
    df = pd.read_csv("data/produk_clean.csv")
    print("Load dari produk_clean.csv")
except FileNotFoundError:
    df = pd.read_csv("data/produk_tokopedia.csv")
    df = preprocess_dataframe(df)
    df.to_csv("data/produk_clean.csv", index=False)

vectorizer, tfidf_matrix = build_index(df)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_result():
    query = request.form.get("query", "").strip()
    top_k = int(request.form.get("top_k", 10))

    results, precision, recall = search(query, df, vectorizer, tfidf_matrix, top_k)

    return render_template("result.html",
        query=query,
        results=results.to_dict(orient="records"),
        precision=round(precision, 4),
        recall=round(recall, 4),
        total=len(results)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=True, port=port)