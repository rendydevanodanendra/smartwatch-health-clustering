import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import os
import warnings

warnings.filterwarnings("ignore")

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Smartwatch Health Clustering",
    page_icon="⌚",
    layout="wide"
)

# Judul Aplikasi
st.title("⌚ Smartwatch Health Data Clustering")
st.write("Aplikasi ini menggunakan Machine Learning untuk mengelompokkan data kesehatan dari smartwatch.")

# --- 1. LOAD DATA ---
# Fungsi untuk memuat data dengan cache agar lebih cepat
@st.cache_data
def load_data():
    # Coba baca file default jika ada di repository
    default_file = "smartwatch_health.csv"
    if os.path.exists(default_file):
        return pd.read_csv(default_file)
    else:
        return None

# Coba muat data default
df = load_data()

# Sidebar untuk Upload jika data default tidak ditemukan atau user ingin ganti
st.sidebar.header("📂 Konfigurasi Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda (Opsional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File kustom berhasil dimuat!")
elif df is not None:
    st.sidebar.info(f"ℹ️ Menggunakan file default: `smartwatch_health.csv`")
else:
    st.warning("⚠️ File `smartwatch_health.csv` tidak ditemukan di repository dan belum ada file yang diupload.")
    st.stop() # Hentikan aplikasi jika tidak ada data sama sekali

# --- 2. PREPROCESSING OTOMATIS ---
# Tampilkan spinner saat memproses data
with st.spinner('🔄 Sedang memproses data...'):
    
    # 2.1. Validasi Kolom Minimal
    required_columns = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count", "Sleep Duration (hours)", "Stress Level"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"❌ File CSV kurang kolom berikut: {', '.join(missing_cols)}")
        st.stop()

    # 2.2. Bersihkan Nama Kolom (hapus spasi di awal/akhir)
    df.columns = df.columns.str.strip()

    # 2.3. Konversi ke Numerik (paksa error jadi NaN)
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2.4. Filter Data Tidak Masuk Akal (PENTING!)
    # Ini mengatasi masalah grafik aneh yang kita lihat sebelumnya
    df_clean = df.copy()
    df_clean = df_clean[
        (df_clean["Heart Rate (BPM)"] > 30) & (df_clean["Heart Rate (BPM)"] < 250) &
        (df_clean["Blood Oxygen Level (%)"] > 50) & (df_clean["Blood Oxygen Level (%)"] <= 100) &
        (df_clean["Step Count"] >= 0) & (df_clean["Step Count"] < 100000) &
        (df_clean["Sleep Duration (hours)"] > 0) & (df_clean["Sleep Duration (hours)"] < 24)
    ]
    
    if len(df_clean) == 0:
        st.error("❌ Semua data terfilter karena nilainya tidak masuk akal. Cek format file CSV Anda.")
        st.stop()

    # 2.5. Imputasi Nilai Kosong (jika ada setelah filtering)
    imputer = SimpleImputer(strategy='median')
    df_clean[required_columns] = imputer.fit_transform(df_clean[required_columns])

    # 2.6. Feature Engineering
    # Tambah sedikit nilai epsilon (0.1) agar tidak dibagi 0
    df_clean["Activity_Score"] = df_clean["Step Count"] / (df_clean["Sleep Duration (hours)"] + 0.1)
    df_clean["Health_Index"] = (df_clean["Blood Oxygen Level (%)"] / (df_clean["Heart Rate (BPM)"] + 0.1)) * 100

    # Fitur final yang akan dipakai clustering
    features_to_use = required_columns + ["Activity_Score", "Health_Index"]

    # 2.7. Scaling & PCA
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_clean[features_to_use])

    # PCA untuk visualisasi 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clean['PC1'] = X_pca[:, 0]
    df_clean['PC2'] = X_pca[:, 1]

# --- 3. INTERFACE UTAMA ---

# Tampilkan Tab
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Data", "⚙️ Clustering", "ℹ️ Tentang"])

with tab1:
    st.subheader("Data Awal (5 Baris Pertama)")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Data Awal", len(df))
    with col2:
        st.metric("Data Bersih (setelah filter)", len(df_clean))
        
    with st.expander("Lihat Statistik Data Bersih"):
        st.write(df_clean[features_to_use].describe())

with tab2:
    st.subheader("Eksperimen Clustering")
    
    # Pilihan Model di Sidebar agar lebih rapi
    model_type = st.sidebar.selectbox("Pilih Algoritma", ["K-Means", "GMM (Gaussian Mixture)"])
    k = st.sidebar.slider("Jumlah Cluster (K)", 2, 6, 3)

    if st.button("🚀 Jalankan Clustering"):
        if model_type == "K-Means":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
        else:
            model = GaussianMixture(n_components=k, random_state=42)
            
        # Prediksi cluster
        clusters = model.fit_predict(X_scaled) # Kita clustering data yang DISCALING, bukan PCA
        df_clean["Cluster"] = clusters
        
        # Hitung skor silhouette
        sil_score = silhouette_score(X_scaled, clusters)
        
        st.success(f"✅ Clustering Selesai! Silhouette Score: **{sil_score:.3f}**")

        # Visualisasi PCA
        st.subheader("Visualisasi Hasil (Proyeksi PCA 2D)")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_clean, palette='viridis', s=60, ax=ax)
        plt.title(f"{model_type} dengan K={k}")
        st.pyplot(fig)

        # Tampilkan Rata-rata per Cluster
        st.subheader("Profil Rata-rata Setiap Cluster")
        st.dataframe(df_clean.groupby("Cluster")[features_to_use].mean().style.background_gradient(cmap='Blues'))

with tab3:
    st.write("Dibuat untuk demonstrasi analisis data kesehatan smartwatch.")
