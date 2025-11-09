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
import warnings

warnings.filterwarnings("ignore")

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Smartwatch Health Clustering",
    page_icon="⌚",
    layout="wide"
)

# Judul dan Deskripsi
st.title("⌚ Smartwatch Health Data Clustering")
st.write("Aplikasi ini mengelompokkan data kesehatan dari smartwatch menggunakan berbagai algoritma machine learning untuk menemukan pola tersembunyi.")

# --- 1. UPLOAD DATA ---
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload file CSV Anda", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("File berhasil diupload!")
else:
    st.info("Menunggu file CSV diupload. Silakan gunakan sidebar.")
    st.stop()

# --- 2. PREPROCESSING (Otomatis) ---
with st.spinner('Sedang memproses data...'):
    # Fitur utama yang diharapkan
    expected_features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                         "Sleep Duration (hours)", "Stress Level", "Activity Level"]
    
    # Validasi kolom
    if not all(col in df.columns for col in expected_features):
        st.error(f"File CSV harus memiliki kolom: {', '.join(expected_features)}")
        st.stop()

    # Encode activity level
    activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)

    # Konversi ke numerik
    numeric_cols = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                    "Sleep Duration (hours)", "Stress Level"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fitur Turunan
    df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
    df["Health_Index"] = (df["Blood Oxygen Level (%)"] / df["Heart Rate (BPM)"]) * 100
    df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
    df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
    # df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1) # Opsional jika terlalu kompleks

    # Final Features untuk Clustering
    features_to_use = numeric_cols + ["Activity_ord", "Activity_Score", "Health_Index", "Recovery_Score", "Fatigue_Index"]
    
    # Imputasi
    imputer = SimpleImputer(strategy='median')
    df_clean = df.copy()
    df_clean[features_to_use] = imputer.fit_transform(df[features_to_use])

    # Scaling & Transformasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_clean[features_to_use])

    pt = PowerTransformer(method='yeo-johnson')
    X_power = pt.fit_transform(X_scaled)

    # PCA
    pca = PCA(n_components=2) # Kita gunakan 2 komponen untuk visualisasi 2D yang mudah
    X_pca = pca.fit_transform(X_power)
    
    # Tambahkan hasil PCA ke dataframe untuk plotting nanti
    df_clean['PC1'] = X_pca[:, 0]
    df_clean['PC2'] = X_pca[:, 1]

# Tampilkan Ringkasan Data
with st.expander("Lihat Data yang Telah Diproses"):
    st.dataframe(df_clean.head())
    st.write(f"Dimensi Data: {df_clean.shape}")

# --- 3. MODEL SELECTION & TUNING ---
st.sidebar.header("2. Konfigurasi Model")

model_option = st.sidebar.selectbox(
    "Pilih Algoritma Clustering",
    ("K-Means", "Gaussian Mixture Model (GMM)", "Spectral Clustering")
)

k_clusters = st.sidebar.slider("Jumlah Cluster (K)", min_value=2, max_value=10, value=3)

# --- 4. CLUSTERING PROCESS ---
if st.sidebar.button("Jalankan Clustering"):
    with st.spinner(f'Menjalankan {model_option} dengan K={k_clusters}...'):
        if model_option == "K-Means":
            model = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            labels = model.fit_predict(X_pca)
        elif model_option == "Gaussian Mixture Model (GMM)":
            model = GaussianMixture(n_components=k_clusters, random_state=42)
            labels = model.fit_predict(X_pca)
        elif model_option == "Spectral Clustering":
            # Spectral bisa berat, kita gunakan subset jika data terlalu besar untuk demo ini
            if len(df_clean) > 5000:
                 st.warning("Spectral Clustering mungkin lambat pada data besar. Mempertimbangkan sampling jika perlu.")
            model = SpectralClustering(n_clusters=k_clusters, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42) # 'nearest_neighbors' sering lebih cepat dari 'rbf'
            labels = model.fit_predict(X_pca)

        # Simpan label ke dataframe
        df_clean['Cluster_Label'] = labels
        
        # Hitung Metrik
        sil_score = silhouette_score(X_pca, labels)

    # --- 5. VISUALISASI HASIL ---
    st.header(f"Hasil Clustering: {model_option}")
    
    # Metrik Utama
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Cluster (K)", k_clusters)
    with col2:
        st.metric("Silhouette Score", f"{sil_score:.3f}")

    # Plot PCA
    st.subheader("Visualisasi Cluster (PCA 2D)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster_Label', data=df_clean, palette='viridis', s=50, ax=ax, legend='full')
    ax.set_title(f"{model_option} Clustering Results on PCA")
    st.pyplot(fig)

    # --- 6. INTERPRETASI CLUSTER ---
    st.header("Interpretasi Cluster")

    # Profiling Rata-rata
    st.subheader("Profil Rata-rata per Cluster")
    cluster_means = df_clean.groupby('Cluster_Label')[features_to_use].mean()
    st.dataframe(cluster_means.style.background_gradient(cmap='Blues', axis=0)) # axis=0 membandingkan antar cluster untuk setiap fitur

    # Box Plot Interaktif
    st.subheader("Distribusi Fitur per Cluster")
    selected_feature = st.selectbox("Pilih Fitur untuk Dianalisis:", features_to_use)
    
    fig_box, ax_box = plt.subplots(figsize=(8, 5))
    sns.boxplot(x='Cluster_Label', y=selected_feature, data=df_clean, palette='viridis', ax=ax_box)
    ax_box.set_title(f"Distribusi {selected_feature} berdasarkan Cluster")
    st.pyplot(fig_box)

    # --- 7. DOWNLOAD HASIL ---
    st.subheader("Download Hasil")
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Data dengan Label Cluster (CSV)",
        data=csv,
        file_name=f'smartwatch_clusters_{model_option}_k{k_clusters}.csv',
        mime='text/csv',
    )

else:
    st.info("Pilih konfigurasi di sidebar dan klik 'Jalankan Clustering' untuk melihat hasil.")