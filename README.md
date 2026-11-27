# Ulas — Aplikasi Analisis Sentimen Ulasan

## Masalah yang diselesaikan
Memilah ribuan ulasan pengguna aplikasi secara manual membutuhkan waktu lama. Aplikasi **Ulas** membantu mengelompokkan teks ulasan berbahasa Indonesia secara otomatis ke dalam sentimen **Positif** atau **Negatif** sehingga masukan pengguna dapat dipantau lebih cepat.

## Dataset
- **Judul & Sumber:** [Gojek App Reviews Indonesia — Kaggle](https://www.kaggle.com/datasets/izzahfathiyyah123/gojek-app-reviews-indonesia)
- **Pembuat Dataset:** Izzah Fathiyyah (`izzahfathiyyah123`)
- **Lisensi:** CC0: Public Domain
- **Jumlah data:** **1.595 ulasan bersih** (854 Negatif dan 741 Positif) setelah pembersihan data duplikat serta penyisihan rating netral (bintang 3).

## Metode
- **Preprocessing:** Pembersihan nilai kosong dan duplikat, pemetaan rating bintang 4–5 menjadi `Positif` dan bintang 1–2 menjadi `Negatif`.
- **Ekstraksi fitur:** `CountVectorizer` (Bag of Words) yang di-*fit* hanya pada data latih agar tidak terjadi kebocoran data uji.
- **Model:** Klasifikasi teks menggunakan `MultinomialNB` (Naive Bayes) dengan pembagian data 75% latih dan 25% uji (`stratify`).
- **Deployment:** Model dan vectorizer disimpan dengan `joblib` lalu disajikan dalam antarmuka web berbasis **Django** (`app_django.py`).

## Hasil & evaluasi
- **Akurasi Data Latih:** 88,9% (0,889)
- **Akurasi Data Uji:** 84,0% (0,840)
- **Confusion Matrix** (Baris = Asli, Kolom = Prediksi `[Positif, Negatif]`):
  - `[[134, 51], [13, 201]]`
  - Dari 214 ulasan Negatif di data uji, 201 berhasil terdeteksi dengan benar (13 lolos sebagai Positif).
  - Dari 185 ulasan Positif di data uji, 134 terdeteksi benar dan 51 salah diklasifikasikan sebagai Negatif.

## Cara menjalankan
1. Instal seluruh pustaka yang dibutuhkan:
   ```bash
   pip install -r requirements.txt
   ```
2. Latih model terlebih dahulu untuk menghasilkan `vectorizer.joblib` dan `model_sentimen.joblib`:
   ```bash
   python3 train.py
   ```
3. Jalankan aplikasi web **Django**:
   ```bash
   python3 app_django.py runserver
   ```
4. Buka browser dan akses **`http://127.0.0.1:8000`**.

## Refleksi singkat
- **Yang berhasil:** Alur lengkap mulai dari pembersihan data, pembagian latih/uji, evaluasi jujur, penyimpanan model dengan `joblib`, hingga integrasi ke antarmuka web Django berjalan mulus tanpa error. Model mampu membedakan ulasan pujian dan keluhan dengan akurasi uji 84,0%.
- **Keterbatasan:** Karena `CountVectorizer` standar memecah kalimat per satu kata (*unigram*), model belum mampu menangkap konteks negasi seperti `"kurang bagus"` (kata `"bagus"` tetap mendominasi sehingga tertebak Positif). Selain itu, kalimat obrolan netral di luar konteks ulasan tetap dipaksa masuk ke salah satu dari dua kelas karena tidak ada kelas Netral.
- **Rencana perbaikan:** Menambahkan fitur *n-gram* (`ngram_range=(1, 2)`), membersihkan kata tidak baku/singkatan (*text normalization*), serta menambahkan kelas Netral atau ambang batas keyakinan (*confidence threshold*).