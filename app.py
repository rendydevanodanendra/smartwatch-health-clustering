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

# --- KONFIGURASI HALAMAN STREAMLIT ---
st.set_page_config(
    page_title="Smartwatch Health Clustering V6",
    page_icon="⌚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- JUDUL APLIKASI ---
st.title("⌚ Smartwatch Health Data Clustering (Final V6)")
st.markdown("""
Aplikasi ini melakukan analisis clustering pada data kesehatan smartwatch untuk menemukan pola tersembunyi.
Menggunakan perbandingan multi-model (K-Means, GMM, Spectral Clustering) dengan preprocessing tingkat lanjut.
""")

# --- 1. LOAD & PREPROCESSING DATA (Di-cache agar cepat) ---
@st.cache_data
def load_and_preprocess_data(uploaded_file=None):
    # 1. Load Data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        default_file = "smartwatch_health.csv"
        if os.path.exists(default_file):
            df = pd.read_csv(default_file)
        else:
            # Jika tidak ada file sama sekali, return None
            return None, None, None, None, None

    # 2. Definisi Fitur Awal
    features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                "Sleep Duration (hours)", "Stress Level"]

    # 3. Encode Activity Level
    # Cek apakah kolom ada, untuk menghindari error jika nama kolom beda
    if "Activity Level" in df.columns:
        activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
        df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
        features.append("Activity_ord")
    elif "Activity_ord" in df.columns: # Jika sudah ter-encode di CSV
         features.append("Activity_ord")

    # 4. Konversi ke Numerik (PENTING: Lakukan sebelum fitur turunan)
    for col in features:
        # Membersihkan karakter non-numerik jika ada (opsional tapi bagus untuk jaga-jaga)
        if col in df.columns and df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Imputasi Awal (untuk fitur dasar sebelum fitur turunan)
    imputer = SimpleImputer(strategy='median')
    # Hanya imputasi kolom yang ada
    valid_features = [f for f in features if f in df.columns]
    df[valid_features] = imputer.fit_transform(df[valid_features])

    # 6. Fitur Turunan (Feature Engineering)
    # Menambahkan konstanta kecil (1e-9) untuk menghindari pembagian dengan nol
    # Pastikan kolom yang dibutuhkan ada sebelum menghitung
    if all(col in df.columns for col in ["Step Count", "Sleep Duration (hours)", "Blood Oxygen Level (%)", "Heart Rate (BPM)", "Stress Level"]):
        df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
        df["Health_Index"] = (df["Blood Oxygen Level (%)"] / (df["Heart Rate (BPM)"] + 1e-9)) * 100
        df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
        df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
        df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)

        # Update daftar fitur yang akan digunakan untuk clustering
        final_features = valid_features + ["Activity_Score", "Health_Index", "Recovery_Score", "Fatigue_Index", "Balance_Score"]
    else:
        final_features = valid_features
        st.warning("Beberapa kolom untuk fitur turunan tidak ditemukan. Menggunakan fitur dasar saja.")

    # Bersihkan NaN/Inf yang mungkin muncul dari fitur turunan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[final_features] = imputer.fit_transform(df[final_features])

    # 7. Scaling & Transformasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[final_features])

    # PowerTransformer kadang gagal jika data terlalu sedikit atau konstan
    try:
        pt = PowerTransformer(method='yeo-johnson')
        X_power = pt.fit_transform(X_scaled)
    except Exception as e:
        st.warning(f"PowerTransformer gagal, menggunakan data terskala RobustScaler: {e}")
        X_power = X_scaled

    X_for_pca = X_power

    # 8. PCA
    pca = PCA(n_components=2) # Gunakan 2 komponen untuk visualisasi mudah
    X_pca = pca.fit_transform(X_for_pca)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    return df, final_features, X_scaled, X_power, X_pca

# --- SIDEBAR: UPLOAD DATA ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda (opsional)", type=["csv"])

# --- LOAD DATA ---
with st.spinner("🔄 Memuat dan memproses data..."):
    df, features, X_scaled, X_power, X_pca = load_and_preprocess_data(uploaded_file)

if df is None:
    st.warning("⚠️ File `smartwatch_health.csv` tidak ditemukan di repository dan Anda belum mengupload file. Silakan upload file CSV di sidebar.")
    st.stop()

st.sidebar.success("✅ Data berhasil dimuat & diproses!")
with st.sidebar.expander("ℹ️ Info Data"):
    st.write(f"Jumlah Baris: {df.shape[0]}")
    st.write(f"Jumlah Fitur: {len(features)}")

# --- TABS UTAMA ---
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Data", "⚙️ Eksperimen Clustering", "💡 Interpretasi & Download"])

# --- TAB 1: RINGKASAN DATA ---
with tab1:
    st.subheader("Data Setelah Preprocessing (5 Baris Pertama)")
    st.dataframe(df.head())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Statistik Deskriptif")
        st.dataframe(df[features].describe())
    with col2:
        st.subheader("Korelasi Fitur")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[features].corr(), annot=False, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# --- TAB 2: EKSPERIMEN CLUSTERING ---
with tab2:
    st.header("🛠️ Konfigurasi & Eksperimen")

    # Pilihan Model & Parameter
    col_model, col_k = st.columns(2)
    with col_model:
        model_name = st.selectbox("Pilih Algoritma Model", ["K-Means (PCA)", "K-Means (Raw Scaled)", "GMM (PCA)", "Spectral Clustering (PCA)"])
    with col_k:
        k_range = st.slider("Rentang Jumlah Cluster (K) untuk Eksperimen", 2, 10, (2, 6))

    # Tombol Jalankan
    if st.button("🚀 Jalankan Eksperimen Otomatis"):
        results = []
        best_score = -1
        best_k = -1
        best_labels = None
        best_model_type = ""

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Loop eksperimen K
        k_values = range(k_range[0], k_range[1] + 1)
        total_steps = len(k_values)

        for i, k in enumerate(k_values):
            status_text.text(f"⏳ Sedang menjalankan K={k}...")
            
            # Pilih data input berdasarkan model
            if "PCA" in model_name:
                X_input = X_pca
            else: # Raw Scaled
                X_input = X_power # Menggunakan hasil PowerTransformer sebagai "Raw Scaled" yang lebih baik

            # Jalankan Model
            if "K-Means" in model_name:
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_input)
            elif "GMM" in model_name:
                model = GaussianMixture(n_components=k, random_state=42)
                labels = model.fit_predict(X_input)
            elif "Spectral" in model_name:
                # Peringatan untuk data besar
                if len(df) > 2000:
                     labels = np.zeros(len(df)) # Placeholder agar tidak crash, idealnya sampling
                     st.warning(f"Spectral Clustering terlalu lambat untuk {len(df)} data. Hasil K={k} dilewati.")
                     sil = -1
                else:
                    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
                    labels = model.fit_predict(X_input)
            
            # Hitung Silhouette Score (jika berhasil)
            if "Spectral" not in model_name or len(df) <= 2000:
                sil = silhouette_score(X_input, labels)
                results.append({"K": k, "Silhouette Score": sil})

                if sil > best_score:
                    best_score = sil
                    best_k = k
                    best_labels = labels
                    best_model_type = model_name

            progress_bar.progress((i + 1) / total_steps)

        progress_bar.empty()
        status_text.text("✅ Eksperimen Selesai!")

        # Simpan hasil terbaik ke session state agar tidak hilang saat refresh
        st.session_state['best_labels'] = best_labels
        st.session_state['best_k'] = best_k
        st.session_state['best_score'] = best_score
        st.session_state['best_model'] = best_model_type
        st.session_state['results_df'] = pd.DataFrame(results)

    # Tampilkan Hasil Eksperimen jika sudah ada di session state
    if 'results_df' in st.session_state:
        st.subheader("📈 Hasil Eksperimen")
        
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            # Highlight nilai maksimum di kolom Silhouette Score
            st.dataframe(st.session_state['results_df'].style.highlight_max(axis=0, color='lightgreen', subset=['Silhouette Score']))
            st.success(f"🏆 K Terbaik: **{st.session_state['best_k']}** (Score: {st.session_state['best_score']:.3f})")
        
        with col_res2:
            # Line chart untuk elbow method/silhouette analysis
            fig_line, ax_line = plt.subplots()
            sns.lineplot(data=st.session_state['results_df'], x="K", y="Silhouette Score", marker="o", ax=ax_line)
            ax_line.set_title("Silhouette Score vs Jumlah Cluster (K)")
            ax_line.set_xlabel("Jumlah Cluster (K)")
            ax_line.set_ylabel("Silhouette Score")
            ax_line.axvline(x=st.session_state['best_k'], color='r', linestyle='--', label=f'Best K={st.session_state["best_k"]}')
            ax_line.legend()
            st.pyplot(fig_line)

        # Visualisasi Cluster Terbaik
        st.subheader(f"Visualisasi Cluster Terbaik (K={st.session_state['best_k']})")
        
        # PENTING: Pastikan panjang labels sama dengan df sebelum assign
        if len(st.session_state['best_labels']) == len(df):
            df['Cluster'] = st.session_state['best_labels'] # Simpan label ke dataframe utama

            fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
            # Gunakan palette yang jelas untuk kategori
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='bright', s=60, ax=ax_scatter, legend='full')
            ax_scatter.set_title(f"Proyeksi PCA - {st.session_state['best_model']} (K={st.session_state['best_k']})")
            st.pyplot(fig_scatter)
        else:
            st.error("Terjadi kesalahan: Jumlah label cluster tidak sesuai dengan jumlah data. Silakan jalankan ulang eksperimen.")

# --- TAB 3: INTERPRETASI & DOWNLOAD ---
with tab3:
    # Cek apakah 'Cluster' sudah ada di df (artinya eksperimen sudah dijalankan)
    if 'Cluster' in df.columns:
        st.header("💡 Interpretasi Hasil")
        
        # 1. Profiling Rata-rata Cluster
        st.subheader("Rata-rata Fitur per Cluster")
        # Hitung rata-rata per cluster
        cluster_means = df.groupby('Cluster')[features].mean()
        # Tampilkan dengan gradasi warna untuk memudahkan melihat nilai tinggi/rendah
        st.dataframe(cluster_means.style.background_gradient(cmap='Blues'))

        # 2. Box Plot Interaktif
        st.subheader("Distribusi Fitur Antar Cluster")
        selected_feature = st.selectbox("Pilih Fitur untuk Dianalisis:", features)
        
        fig_box, ax_box = plt.subplots(figsize=(8, 4))
        sns.boxplot(x='Cluster', y=selected_feature, data=df, palette='viridis', ax=ax_box)
        ax_box.set_title(f"Distribusi {selected_feature} per Cluster")
        st.pyplot(fig_box)

        # 3. Download Data Hasil
        st.subheader("📥 Download Hasil")
        # Konversi dataframe ke CSV untuk didownload
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Data dengan Label Cluster (CSV)",
            data=csv,
            file_name=f'smartwatch_clustering_results_K{st.session_state["best_k"]}.csv',
            mime='text/csv',
        )
    else:
        st.info("⚠️ Silakan jalankan eksperimen di Tab 2 terlebih dahulu untuk melihat interpretasi.")
