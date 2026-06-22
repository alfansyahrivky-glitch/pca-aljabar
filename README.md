# PCA Face Similarity Verification - Streamlit

Aplikasi ini membandingkan foto wajah lama dan foto wajah sekarang menggunakan PCA dan cosine similarity.

## Fitur

- Upload foto lama dan foto sekarang.
- Deteksi wajah otomatis menggunakan OpenCV Haar Cascade.
- Crop wajah otomatis.
- Ekstraksi fitur menggunakan PCA.
- Perhitungan cosine similarity.
- Konversi skor menjadi estimasi persentase kemiripan.
- Mode satu foto lama vs banyak foto sekarang.

## Cara Menjalankan di Lokal

1. Install dependency:

```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:

```bash
streamlit run app.py
```

3. Buka browser sesuai link yang muncul dari Streamlit.

> Catatan: Dataset tidak disertakan di repository. Dataset LFW akan diunduh secara otomatis oleh `scikit-learn` saat aplikasi dijalankan.

## Cara Menjalankan di Streamlit Community Cloud

1. Upload file berikut ke GitHub:
   - `app.py`
   - `requirements.txt`
   - `README.md`

2. Buka Streamlit Community Cloud.
3. Pilih repository.
4. Set main file path ke:

```text
app.py
```

5. Deploy.

## Catatan

Sistem ini adalah demo sederhana berbasis PCA. Hasil persentase bukan bukti identitas mutlak.
Untuk aplikasi verifikasi wajah yang serius, gunakan model modern seperti FaceNet, ArcFace, DeepFace, atau Dlib Face Recognition.
