import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Smartwatch Health Clustering", layout="wide")
st.title("⌚ Smartwatch Health Clustering Dashboard")

# Upload CSV
uploaded = st.file_uploader("📁 Upload smartwatch_health.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

    # Pilih kolom utama
    features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                "Sleep Duration (hours)", "Stress Level"]
    activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
    features.append("Activity_ord")

    # Tambahkan fitur turunan
    df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
    df["Health_Index"] = (df["Blood Oxygen Level (%)"] / df["Heart Rate (BPM)"]) * 100
    df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
    df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
    df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)
    features += ["Activity_Score", "Health_Index", "Recovery_Score", "Fatigue_Index", "Balance_Score"]

    # Bersihkan data
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy="median")
    df[features] = imputer.fit_transform(df[features])

    # Scaling dan normalisasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    pt = PowerTransformer(method="yeo-johnson")
    X_power = pt.fit_transform(X_scaled)
    X_log = np.log1p(np.abs(X_power))
    X_log = np.nan_to_num(X_log, nan=0.0)

    # PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_log)

    # Clustering
    sil_scores = {}
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=25)
        labels = kmeans.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        sil_scores[k] = sil

    best_k = max(sil_scores, key=sil_scores.get)
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=25)
    df["Cluster"] = kmeans.fit_predict(X_pca)
    best_sil = sil_scores[best_k]

    # Output utama
    st.subheader("📈 Hasil Clustering")
    st.write(f"**Cluster terbaik:** K = {best_k} | **Silhouette Score = {best_sil:.3f}**")

    # Visualisasi PCA
    fig, ax = plt.subplots(figsize=(8,6))
    sc = ax.scatter(X_pca[:,0], X_pca[:,1], c=df["Cluster"], cmap="rainbow", s=10)
    plt.title(f"KMeans Clusters (PCA Projection) — K={best_k} | Silhouette={best_sil:.3f}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(alpha=0.3)
    plt.colorbar(sc)
    st.pyplot(fig)

    # Tabel ringkasan cluster
    summary = df.groupby("Cluster")[features].mean().round(2)
    st.subheader("📊 Rata-Rata Setiap Cluster")
    st.dataframe(summary)

    # Insight otomatis
    st.subheader("🧠 Insight Otomatis per Cluster")
    for i, row in summary.iterrows():
        desc = []
        if row["Step Count"] > summary["Step Count"].mean():
            desc.append("aktif")
        else:
            desc.append("kurang aktif")

        if row["Sleep Duration (hours)"] < summary["Sleep Duration (hours)"].mean():
            desc.append("kurang tidur")

        if row["Stress Level"] > summary["Stress Level"].mean():
            desc.append("stres tinggi")
        elif row["Stress Level"] < summary["Stress Level"].mean():
            desc.append("relatif tenang")

        st.write(f"- **Cluster {i}** → HR={row['Heart Rate (BPM)']:.1f}, BO={row['Blood Oxygen Level (%)']:.1f}, "
                 f"Step={row['Step Count']:.0f} → {' & '.join(desc)}")

    st.success("✅ Analisis selesai — insight otomatis ditampilkan!")

else:
    st.info("⬆️ Silakan upload file smartwatch_health.csv terlebih dahulu.")
