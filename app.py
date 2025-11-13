# =============================================
# SMARTWATCH HEALTH CLUSTERING — STREAMLIT APP
# =============================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# =============================================
# 🧱 TITLE DAN DESKRIPSI
# =============================================
st.title("⌚ Smartwatch Health Clustering Dashboard")
st.markdown("""
Aplikasi ini menggunakan data sensor wearable (heart rate, sleep, stress, dsb)  
untuk **menemukan pola kesehatan tersembunyi** dengan teknik *unsupervised learning*.
""")

# =============================================
# 📂 UPLOAD DATA
# =============================================
uploaded_file = st.file_uploader("Upload file CSV data smartwatch kamu", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # tampilkan beberapa data awal
    st.subheader("📋 Data Awal")
    st.dataframe(df.head())

    # =============================================
    # 🔹 PREPROCESSING
    # =============================================
    st.subheader("⚙️ Preprocessing Data")

    features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                "Sleep Duration (hours)", "Stress Level"]

    # Encode aktivitas
    activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
    features.append("Activity_ord")

    # Pastikan numerik
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])

    # Scaling dan transformasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])

    pt = PowerTransformer(method='yeo-johnson')
    X_power = pt.fit_transform(X_scaled)

    # PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_power)

    st.success("Data berhasil diproses dan siap untuk clustering!")

    # =============================================
    # ⚙️ PILIH MODEL
    # =============================================
    st.subheader("🤖 Pilih Model Clustering")
    model_name = st.selectbox("Pilih model:", ["KMeans", "Gaussian Mixture (GMM)", "Spectral Clustering"])

    k = st.slider("Jumlah Cluster (K)", 2, 10, 3)

    # =============================================
    # 🔹 CLUSTERING
    # =============================================
    if st.button("Jalankan Clustering"):
        if model_name == "KMeans":
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
        elif model_name == "Gaussian Mixture (GMM)":
            model = GaussianMixture(n_components=k, random_state=42)
        else:
            model = SpectralClustering(n_clusters=k, affinity='rbf', assign_labels='kmeans', random_state=42)

        labels = model.fit_predict(X_pca)
        df["Cluster"] = labels

        sil = silhouette_score(X_pca, labels)
        st.metric("Nilai Silhouette", f"{sil:.3f}")

        # =============================================
        # 🔹 VISUALISASI CLUSTER
        # =============================================
        st.subheader("📊 Visualisasi Cluster (PCA 2D)")
        fig, ax = plt.subplots(figsize=(7,5))
        scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="rainbow", s=20)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model_name} — K={k}, Silhouette={sil:.3f}")
        st.pyplot(fig)

        # =============================================
        # 🔹 TABEL HASIL
        # =============================================
        st.subheader("📈 Hasil Cluster")
        st.dataframe(df.head())

        # =============================================
        # 💾 DOWNLOAD
        # =============================================
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Hasil CSV",
            data=csv,
            file_name='smartwatch_cluster_result.csv',
            mime='text/csv',
        )
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
