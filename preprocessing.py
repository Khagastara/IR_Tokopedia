import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stopword_remover = stop_factory.create_stop_word_remover()

def preprocess(text):
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def preprocess_dataframe(df):
    print("Preprocessing data...")

    df = df.copy()
    df = df.dropna(subset=["product_name"])
    df = df.drop_duplicates(subset=["product_name"])
    df = df.reset_index(drop=True)

    # Gunakan product_name + keyword sebagai dokumen teks utama
    df["document"] = (
        df["product_name"].fillna("") + " " +
        df["keyword"].fillna("")
    )

    df["document_clean"] = df["document"].apply(preprocess)

    df = df[df["document_clean"].str.strip() != ""]
    df = df.reset_index(drop=True)

    print(f"Total dokumen siap: {len(df)}")
    return df