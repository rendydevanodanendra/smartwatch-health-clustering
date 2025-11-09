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

st.set_page_config(page_title="Smartwatch Health Clustering", layout="wide")

st.title("⌚ Smartwatch Health Clustering Dashboard")
st.markdown("### 📊 Analisis Kesehatan Pengguna Berdasarkan Data Smartwatch")

# =============================================
# UPLOAD DATA
# =============================================
uploaded_file = st.file_uploader("📂 Upload file CSV data smartwatch", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # 1️⃣ Fitur utama
    features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                "Sleep Duration (hours)", "Stress Level"]

    # 2️⃣ Encode activity level
    activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
    features.append("Activity_ord")

    # 3️⃣ Pastikan numerik
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    imputer = SimpleImputer(strategy='median')
    df[features] = imputer.fit_transform(df[features])

    # 4️⃣ Transformasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[features])
    pt = PowerTransformer(method='yeo-johnson')
    X_power = pt.fit_transform(X_scaled)
    X_log = np.log1p(np.abs(X_power))
    X_log = np.nan_to_num(X_log, nan=0.0)

    # PCA
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_log)

    # =============================================
    # CLUSTERING
    # =============================================
    models = {
        "KMeans+PCA": KMeans,
        "Spectral": SpectralClustering,
        "GMM": GaussianMixture
    }

    results = []
    for model_name, model_class in models.items():
        sil_scores = {}
        for k in range(2, 10):
            if model_name == "KMeans+PCA":
                model = model_class(n_clusters=k, random_state=42, n_init=20)
                labels = model.fit_predict(X_pca)
            elif model_name == "GMM":
                model = model_class(n_components=k, random_state=42)
                labels = model.fit_predict(X_pca)
            else:  # Spectral
                model = model_class(n_clusters=k, affinity='rbf', assign_labels='kmeans', random_state=42)
                labels = model.fit_predict(X_pca)
            sil = silhouette_score(X_pca, labels)
            sil_scores[k] = sil

        best_k = max(sil_scores, key=sil_scores.get)
        results.append({
            "Model": model_name,
            "Best K": best_k,
            "Silhouette": sil_scores[best_k]
        })

    # =============================================
    # HASIL
    # =============================================
    st.subheader("📈 Hasil Evaluasi Model")
    results_df = pd.DataFrame(results).sort_values(by="Silhouette", ascending=False)
    st.dataframe(results_df, use_container_width=True)

    best_model = results_df.iloc[0]
    st.success(f"🔥 Model terbaik: **{best_model['Model']} (K={best_model['Best K']})** "
               f"dengan Silhouette = **{best_model['Silhouette']:.3f}**")

    # =============================================
    # VISUALISASI CLUSTER
    # =============================================
    if best_model['Model'] == "KMeans+PCA":
        best_model_fit = KMeans(n_clusters=int(best_model["Best K"]), random_state=42, n_init=20).fit(X_pca)
        labels = best_model_fit.labels_
    elif best_model['Model'] == "GMM":
        best_model_fit = GaussianMixture(n_components=int(best_model["Best K"]), random_state=42).fit(X_pca)
        labels = best_model_fit.predict(X_pca)
    else:
        best_model_fit = SpectralClustering(n_clusters=int(best_model["Best K"]),
                                            affinity='rbf', assign_labels='kmeans', random_state=42)
        labels = best_model_fit.fit_predict(X_pca)

    df["Cluster"] = labels

    st.subheader("🎯 Visualisasi Cluster")
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap="rainbow", s=15)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"{best_model['Model']} (K={best_model['Best K']}) — Silhouette={best_model['Silhouette']:.3f}")
    st.pyplot(fig)

    # =============================================
    # INSIGHT OTOMATIS
    # =============================================
    st.subheader("🧠 Insight Otomatis per Cluster")
    summary = df.groupby("Cluster")[features].mean().round(2)
    st.dataframe(summary, use_container_width=True)

    st.markdown("### 💬 Interpretasi Sederhana")
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
        else:
            desc.append("relatif tenang")

        st.write(f"- Cluster **{i}** → HR={row['Heart Rate (BPM)']:.1f}, BO={row['Blood Oxygen Level (%)']:.1f}, "
                 f"Step={row['Step Count']:.0f} → {' & '.join(desc)}")

    # =============================================
    # EXPORT HASIL
    # =============================================
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download hasil clustering (CSV)", csv, "smartwatch_cluster_result.csv", "text/csv")
