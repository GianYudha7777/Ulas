import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def siapkan_data(nama_file):
    data = pd.read_csv(nama_file)
    data = data[["content", "score"]].dropna().drop_duplicates()
    data = data[data["score"] != 3].copy()
    data["sentimen"] = data["score"].apply(
        lambda rating: "Positif" if rating >= 4 else "Negatif"
    )
    return data


def latih_dan_evaluasi(data):
    teks_latih, teks_uji, label_latih, label_uji = train_test_split(
        data["content"],
        data["sentimen"],
        test_size=0.25,
        random_state=42,
        stratify=data["sentimen"],
    )

    vectorizer = CountVectorizer(min_df=3)
    fitur_latih = vectorizer.fit_transform(teks_latih)
    fitur_uji = vectorizer.transform(teks_uji)

    model = MultinomialNB()
    model.fit(fitur_latih, label_latih)

    akurasi_latih = model.score(fitur_latih, label_latih)
    prediksi = model.predict(fitur_uji)
    akurasi_uji = accuracy_score(label_uji, prediksi)
    matriks = confusion_matrix(label_uji, prediksi, labels=["Positif", "Negatif"])

    print(f"Total data bersih : {len(data)}")
    print(data["sentimen"].value_counts().to_string())
    print(f"\nAkurasi Latih     : {akurasi_latih:.3f} ({akurasi_latih * 100:.1f}%)")
    print(f"Akurasi Uji       : {akurasi_uji:.3f} ({akurasi_uji * 100:.1f}%)")
    print("\nConfusion Matrix [Positif, Negatif]:")
    print(matriks)

    return vectorizer, model


def simpan_model(vectorizer, model):
    joblib.dump(vectorizer, "vectorizer.joblib")
    joblib.dump(model, "model_sentimen.joblib")
    print("\nModel berhasil disimpan.")


if __name__ == "__main__":
    dataset = siapkan_data("gojek_reviews.csv")
    vectorizer, model = latih_dan_evaluasi(dataset)
    simpan_model(vectorizer, model)