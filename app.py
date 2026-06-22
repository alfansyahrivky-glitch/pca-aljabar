import cv2
import numpy as np
import streamlit as st

from PIL import Image
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="PCA Face Similarity Verification",
    layout="wide"
)


@st.cache_resource
def load_and_train_pca(n_components=150, face_size=(64, 64), min_faces_per_person=20):
    """
    Load dataset LFW dari scikit-learn dan latih PCA.
    """

    lfw = fetch_lfw_people(
        min_faces_per_person=min_faces_per_person,
        resize=0.5,
        color=False
    )

    images = lfw.images
    dataset_faces = []

    for img in images:
        resized = cv2.resize(img, face_size)
        resized = resized.astype("float32")

        if resized.max() > 1:
            resized = resized / 255.0

        dataset_faces.append(resized)

    dataset_faces = np.array(dataset_faces)
    x_flat = dataset_faces.reshape(dataset_faces.shape[0], -1)

    max_components = min(n_components, x_flat.shape[0], x_flat.shape[1])

    pca = PCA(
        n_components=max_components,
        whiten=True,
        random_state=42
    )

    pca.fit(x_flat)

    metadata = {
        "jumlah_gambar": images.shape[0],
        "ukuran_wajah": face_size,
        "dimensi_awal": x_flat.shape[1],
        "dimensi_pca": max_components,
        "explained_variance": float(np.sum(pca.explained_variance_ratio_) * 100)
    }

    return pca, metadata


def uploaded_file_to_cv2(uploaded_file):
    """
    Mengubah file upload Streamlit menjadi format gambar OpenCV BGR.
    """

    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    return image_bgr


def detect_and_crop_face_from_bgr(img_bgr, target_size=(64, 64)):
    """
    Deteksi wajah terbesar dari gambar BGR, crop, resize, dan normalisasi.
    """

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    if len(faces) == 0:
        raise ValueError(
            "Wajah tidak terdeteksi. Gunakan foto yang lebih jelas, terang, dan menghadap depan."
        )

    faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, target_size)
    face_normalized = face_resized.astype("float32") / 255.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    preview = img_rgb.copy()
    cv2.rectangle(preview, (x, y), (x+w, y+h), (255, 0, 0), 3)

    return face_normalized, preview


def get_pca_embedding(uploaded_file, pca, face_size=(64, 64)):
    """
    Mengambil embedding PCA dari file gambar upload.
    """

    img_bgr = uploaded_file_to_cv2(uploaded_file)
    face, preview = detect_and_crop_face_from_bgr(img_bgr, target_size=face_size)

    flat = face.reshape(1, -1)
    embedding = pca.transform(flat)

    return embedding, face, preview


def similarity_to_percentage(score):
    """
    Konversi cosine similarity ke persentase estimasi.
    """

    score = float(np.clip(score, -1, 1))
    percentage = ((score + 1) / 2) * 100

    return percentage


def interpret_similarity(percentage):
    """
    Interpretasi sederhana dari hasil kemiripan.
    """

    if percentage >= 80:
        return "Kemiripan sangat tinggi. Kemungkinan orang yang sama cukup kuat, tetapi tetap perlu validasi tambahan."
    elif percentage >= 70:
        return "Kemiripan tinggi. Ada indikasi wajah mirip, tetapi jangan dijadikan satu-satunya dasar keputusan."
    elif percentage >= 60:
        return "Kemiripan sedang. Mungkin mirip, tetapi butuh lebih banyak foto pembanding."
    elif percentage >= 50:
        return "Kemiripan rendah-sedang. Hasil belum cukup kuat."
    else:
        return "Kemiripan rendah. Kemungkinan berbeda, atau kualitas/angle foto kurang mendukung."


def pca_compress(image_matrix, k):
    """
    Kompres satu channel gambar menggunakan PCA.
    """
    mean = np.mean(image_matrix, axis=0)
    centered = image_matrix - mean

    cov_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_index]
    eigenvector_subset = eigenvectors[:, :k]

    compressed = np.dot(centered, eigenvector_subset)
    reconstructed = np.dot(compressed, eigenvector_subset.T) + mean

    return reconstructed


def calculate_compression_ratio(rows, cols, k, channels):
    """
    Hitung persentase ukuran kompresi terhadap ukuran asli.
    """
    original_size = rows * cols * channels
    compressed_size = (rows * k + k * cols) * channels

    return (compressed_size / original_size) * 100


def compress_image_with_pca(image, k):
    """
    Kompres gambar RGB dan grayscale menggunakan PCA.
    """
    img_rgb = image.convert("RGB")
    img_rgb_array = np.array(img_rgb, dtype=float)
    rows, cols, _ = img_rgb_array.shape

    R = img_rgb_array[:, :, 0]
    G = img_rgb_array[:, :, 1]
    B = img_rgb_array[:, :, 2]

    R_compressed = pca_compress(R, k)
    G_compressed = pca_compress(G, k)
    B_compressed = pca_compress(B, k)

    rgb_compressed = np.stack((R_compressed, G_compressed, B_compressed), axis=2)
    rgb_compressed = np.clip(rgb_compressed, 0, 255).astype(np.uint8)

    img_gray = image.convert("L")
    gray_array = np.array(img_gray, dtype=float)
    gray_compressed = pca_compress(gray_array, k)
    gray_compressed = np.clip(gray_compressed, 0, 255).astype(np.uint8)

    original_rgb = img_rgb_array.astype(np.uint8)
    original_gray = np.array(img_gray, dtype=np.uint8)

    ratio_rgb = calculate_compression_ratio(rows, cols, k, 3)
    ratio_gray = calculate_compression_ratio(rows, cols, k, 1)

    return {
        "original_rgb": original_rgb,
        "rgb_compressed": rgb_compressed,
        "original_gray": original_gray,
        "gray_compressed": gray_compressed,
        "ratio_rgb": ratio_rgb,
        "ratio_gray": ratio_gray,
        "rows": rows,
        "cols": cols,
    }


def compare_two_faces(old_file, current_file, pca, face_size=(64, 64)):
    """
    Membandingkan dua foto wajah.
    """

    old_emb, old_face, old_preview = get_pca_embedding(old_file, pca, face_size)
    current_emb, current_face, current_preview = get_pca_embedding(current_file, pca, face_size)

    score = cosine_similarity(old_emb, current_emb)[0][0]
    percentage = similarity_to_percentage(score)

    return {
        "score": float(score),
        "percentage": percentage,
        "interpretation": interpret_similarity(percentage),
        "old_face": old_face,
        "current_face": current_face,
        "old_preview": old_preview,
        "current_preview": current_preview
    }


def compare_one_to_many(old_file, current_files, pca, face_size=(64, 64)):
    """
    Membandingkan satu foto lama dengan banyak foto sekarang.
    """

    old_emb, old_face, old_preview = get_pca_embedding(old_file, pca, face_size)

    scores = []
    percentages = []
    current_faces = []
    current_previews = []
    filenames = []

    for file in current_files:
        current_emb, current_face, current_preview = get_pca_embedding(file, pca, face_size)

        score = cosine_similarity(old_emb, current_emb)[0][0]
        percentage = similarity_to_percentage(score)

        scores.append(float(score))
        percentages.append(float(percentage))
        current_faces.append(current_face)
        current_previews.append(current_preview)
        filenames.append(file.name)

    avg_score = float(np.mean(scores))
    avg_percentage = similarity_to_percentage(avg_score)

    return {
        "scores": scores,
        "percentages": percentages,
        "average_score": avg_score,
        "average_percentage": avg_percentage,
        "interpretation": interpret_similarity(avg_percentage),
        "old_face": old_face,
        "old_preview": old_preview,
        "current_faces": current_faces,
        "current_previews": current_previews,
        "filenames": filenames
    }


st.title("Sistem Verifikasi dan Kompresi Gambar PCA")
st.caption("Aplikasi Streamlit untuk verifikasi wajah dan kompresi gambar menggunakan PCA.")

with st.expander("Penjelasan singkat sistem"):
    st.write(
        """
        Aplikasi ini menyediakan dua fitur:

        1. Verifikasi wajah dengan PCA + cosine similarity.
        2. Kompresi gambar RGB dan grayscale menggunakan PCA.

        Cocok untuk demo computer vision, tugas, dan eksperimen PCA sederhana.
        """
    )

app_mode = st.sidebar.radio(
    "Pilih fitur aplikasi",
    options=[
        "Verifikasi wajah",
        "Kompresi gambar PCA"
    ]
)

if app_mode == "Verifikasi wajah":
    st.sidebar.title("Pengaturan Model")

    n_components = st.sidebar.slider(
        "Jumlah komponen PCA",
        min_value=50,
        max_value=250,
        value=150,
        step=10
    )

    face_size_value = st.sidebar.selectbox(
        "Ukuran wajah",
        options=[48, 64, 80, 96],
        index=1
    )

    min_faces = st.sidebar.selectbox(
        "Minimum foto per orang pada dataset LFW",
        options=[10, 20, 30, 50],
        index=1
    )

    face_size = (face_size_value, face_size_value)

    st.sidebar.info(
        "Semakin besar komponen PCA dan ukuran wajah, semakin berat prosesnya. "
        "Kalau laptop mulai mengeluh, kecilkan parameternya."
    )

    with st.spinner("Melatih PCA menggunakan dataset LFW..."):
        pca, metadata = load_and_train_pca(
            n_components=n_components,
            face_size=face_size,
            min_faces_per_person=min_faces
        )

    col_meta_1, col_meta_2, col_meta_3, col_meta_4 = st.columns(4)

    col_meta_1.metric("Jumlah Dataset", metadata["jumlah_gambar"])
    col_meta_2.metric("Dimensi Awal", metadata["dimensi_awal"])
    col_meta_3.metric("Dimensi PCA", metadata["dimensi_pca"])
    col_meta_4.metric("Explained Variance", f"{metadata['explained_variance']:.2f}%")

    st.divider()

    mode = st.radio(
        "Pilih mode perbandingan",
        options=[
            "Dua foto: foto lama vs foto sekarang",
            "Satu foto lama vs banyak foto sekarang"
        ]
    )

    st.warning(
        "Catatan: Hasil persentase adalah estimasi kemiripan berbasis PCA, bukan bukti identitas mutlak."
    )

    if mode == "Dua foto: foto lama vs foto sekarang":
        st.subheader("Upload Dua Foto")

        col1, col2 = st.columns(2)

        with col1:
            old_file = st.file_uploader(
                "Upload foto lama, misalnya umur 10 tahun",
                type=["jpg", "jpeg", "png"],
                key="old_single"
            )

        with col2:
            current_file = st.file_uploader(
                "Upload foto sekarang, misalnya umur 16 tahun",
                type=["jpg", "jpeg", "png"],
                key="current_single"
            )

        if old_file is not None and current_file is not None:
            if st.button("Hitung Kemiripan", type="primary"):
                try:
                    result = compare_two_faces(old_file, current_file, pca, face_size)

                    st.success("Perhitungan selesai.")

                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Cosine Similarity", f"{result['score']:.4f}")
                    metric_col2.metric("Estimasi Kemiripan", f"{result['percentage']:.2f}%")

                    st.info(result["interpretation"])

                    st.subheader("Preview Deteksi Wajah")
                    prev_col1, prev_col2 = st.columns(2)
                    prev_col1.image(result["old_preview"], caption="Deteksi wajah pada foto lama", use_container_width=True)
                    prev_col2.image(result["current_preview"], caption="Deteksi wajah pada foto sekarang", use_container_width=True)

                    st.subheader("Wajah Hasil Crop")
                    crop_col1, crop_col2 = st.columns(2)
                    crop_col1.image(result["old_face"], caption="Crop wajah foto lama", clamp=True, use_container_width=True)
                    crop_col2.image(result["current_face"], caption="Crop wajah foto sekarang", clamp=True, use_container_width=True)

                except Exception as error:
                    st.error(str(error))

    else:
        st.subheader("Upload Satu Foto Lama dan Banyak Foto Sekarang")

        old_file = st.file_uploader(
            "Upload satu foto lama",
            type=["jpg", "jpeg", "png"],
            key="old_many"
        )

        current_files = st.file_uploader(
            "Upload beberapa foto sekarang",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="current_many"
        )

        if old_file is not None and current_files:
            if st.button("Hitung Rata-rata Kemiripan", type="primary"):
                try:
                    result = compare_one_to_many(old_file, current_files, pca, face_size)

                    st.success("Perhitungan selesai.")

                    metric_col1, metric_col2 = st.columns(2)
                    metric_col1.metric("Rata-rata Cosine Similarity", f"{result['average_score']:.4f}")
                    metric_col2.metric("Rata-rata Estimasi Kemiripan", f"{result['average_percentage']:.2f}%")

                    st.info(result["interpretation"])

                    st.subheader("Skor Tiap Foto")
                    table_data = []
                    for filename, score, percentage in zip(
                        result["filenames"],
                        result["scores"],
                        result["percentages"]
                    ):
                        table_data.append(
                            {
                                "Nama File": filename,
                                "Cosine Similarity": round(score, 4),
                                "Estimasi Kemiripan": f"{percentage:.2f}%"
                            }
                        )

                    st.dataframe(table_data, use_container_width=True)

                    st.subheader("Foto Lama")
                    st.image(result["old_preview"], caption="Deteksi wajah pada foto lama", use_container_width=True)

                    st.subheader("Hasil Perbandingan Foto Sekarang")
                    for idx, filename in enumerate(result["filenames"]):
                        st.markdown(f"### {idx + 1}. {filename}")
                        st.write(f"Cosine Similarity: `{result['scores'][idx]:.4f}`")
                        st.write(f"Estimasi Kemiripan: `{result['percentages'][idx]:.2f}%`")

                        img_col1, img_col2 = st.columns(2)
                        img_col1.image(
                            result["current_previews"][idx],
                            caption="Deteksi wajah",
                            use_container_width=True
                        )
                        img_col2.image(
                            result["current_faces"][idx],
                            caption="Crop wajah",
                            clamp=True,
                            use_container_width=True
                        )

                except Exception as error:
                    st.error(str(error))

else:
    st.sidebar.title("Pengaturan Kompresi")
    k = st.sidebar.slider(
        "Nilai K untuk kompresi PCA",
        min_value=1,
        max_value=100,
        value=20,
        step=1
    )

    st.subheader("Upload Gambar untuk Kompresi")
    uploaded_image = st.file_uploader(
        "Upload gambar JPG/JPEG/PNG",
        type=["jpg", "jpeg", "png"],
        key="compress_image"
    )

    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            result = compress_image_with_pca(image, k)

            st.success("Kompresi gambar selesai.")

            st.markdown("### Perbandingan Gambar RGB")
            rgb_col1, rgb_col2 = st.columns(2)
            rgb_col1.image(result["original_rgb"], caption="Gambar RGB asli", use_container_width=True)
            rgb_col2.image(result["rgb_compressed"], caption=f"RGB terkompresi (K={k})", use_container_width=True)

            st.markdown("### Perbandingan Gambar Grayscale")
            gray_col1, gray_col2 = st.columns(2)
            gray_col1.image(result["original_gray"], caption="Grayscale asli", clamp=True, use_container_width=True)
            gray_col2.image(result["gray_compressed"], caption=f"Grayscale terkompresi (K={k})", clamp=True, use_container_width=True)

            st.markdown("### Informasi Kompresi")
            st.write(f"Ukuran gambar: {result['rows']} x {result['cols']}")
            st.write(f"Persentase ukuran RGB setelah kompresi: {result['ratio_rgb']:.2f}%")
            st.write(f"Persentase ukuran grayscale setelah kompresi: {result['ratio_gray']:.2f}%")

        except Exception as error:
            st.error(str(error))

st.divider()

st.markdown(
    """
    ### Kesimpulan

    Aplikasi ini sekarang berisi dua mode:

    - Verifikasi wajah dengan PCA dan cosine similarity.
    - Kompresi gambar RGB dan grayscale menggunakan PCA.

    Mode verifikasi cocok untuk demo dan eksperimen.
    Mode kompresi cocok untuk memahami bagaimana PCA dapat merekonstruksi gambar.
    """
)
