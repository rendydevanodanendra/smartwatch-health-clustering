# SMARTWATCH HEALTH CLUSTERING — Streamlit (pipeline sama persis dengan Colab)
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
import sklearn
import warnings
warnings.filterwarnings("ignore")

# --- Setup dasar
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
st.set_page_config(page_title="Smartwatch Health Clustering", layout="wide")

st.title("⌚ Smartwatch Health Clustering (Colab-like Pipeline)")
st.markdown("Upload CSV, pipeline akan dijalankan **sama persis seperti di Colab**: \
impute → robust scale → yeo-johnson → log → PCA → clustering → evaluasi model.")

# --- Sidebar info
st.sidebar.subheader("Environment Info")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"pandas: {pd.__version__}")
st.sidebar.write(f"numpy: {np.__version__}")

# --- Upload file
uploaded_file = st.file_uploader("📁 Upload file CSV (smartwatch_health.csv)", type=["csv"])
if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview Data")
st.dataframe(df.head())

# --- Setup fitur
features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
            "Sleep Duration (hours)", "Stress Level"]

if "Activity Level" in df.columns:
    mapping = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(mapping).fillna(1)
else:
    df["Activity_ord"] = 1
features.append("Activity_ord")

# pastikan numeric
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

# --- Buat fitur turunan
df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
df["Health_Index"] = (df["Blood Oxygen Level (%)"] / df["Heart Rate (BPM)"]) * 100
df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)

st.write("Fitur yang digunakan:")
st.write(features)

# --- Imputasi, scaling, transformasi
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[features])

pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X_scaled)

X_log = np.log1p(np.abs(X_power))
X_log = np.nan_to_num(X_log, nan=0.0)

pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_log)

st.success("✅ Preprocessing selesai (median impute → RobustScaler → Yeo-Johnson → log → PCA)")

# --- Jalankan model & bandingkan
st.subheader("🔍 Perbandingan Model Clustering")

def compute_best_model(name, model_func, X):
    sil_scores = {}
    for k in range(2, 10):
        try:
            labels = model_func(k)
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan
        sil_scores[k] = sil
        st.write(f"[{name}] K={k}, Silhouette={sil:.3f}")
    best_k = max(sil_scores, key=sil_scores.get)
    return best_k, sil_scores[best_k]

# KMeans PCA
best_k_pca, best_sil_pca = compute_best_model(
    "KMeans+PCA",
    lambda k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_pca),
    X_pca
)
df["KMeans_PCA_Label"] = KMeans(n_clusters=best_k_pca, random_state=RANDOM_STATE, n_init=20).fit_predict(X_pca)

# KMeans RAW
best_k_raw, best_sil_raw = compute_best_model(
    "KMeans RAW",
    lambda k: KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit_predict(X_log),
    X_log
)
df["KMeans_Raw_Label"] = KMeans(n_clusters=best_k_raw, random_state=RANDOM_STATE, n_init=20).fit_predict(X_log)

# GMM
best_k_gmm, best_sil_gmm = compute_best_model(
    "GMM",
    lambda k: GaussianMixture(n_components=k, random_state=RANDOM_STATE).fit_predict(X_pca),
    X_pca
)
df["GMM_Label"] = GaussianMixture(n_components=best_k_gmm, random_state=RANDOM_STATE).fit(X_pca).predict(X_pca)

# Spectral
best_k_spec, best_sil_spec = compute_best_model(
    "Spectral",
    lambda k: SpectralClustering(n_clusters=k, affinity='rbf', assign_labels='kmeans', random_state=RANDOM_STATE).fit_predict(X_pca),
    X_pca
)
df["Spectral_Label"] = SpectralClustering(n_clusters=best_k_spec, affinity='rbf', assign_labels='kmeans', random_state=RANDOM_STATE).fit_predict(X_pca)

# --- Hasil akhir
results = pd.DataFrame({
    "Model": ["KMeans+PCA", "KMeans RAW", "GMM", "Spectral"],
    "Best K": [best_k_pca, best_k_raw, best_k_gmm, best_k_spec],
    "Silhouette": [best_sil_pca, best_sil_raw, best_sil_gmm, best_sil_spec]
}).sort_values(by="Silhouette", ascending=False).reset_index(drop=True)

st.subheader("🔥 Hasil Perbandingan Akhir")
st.dataframe(results)

# --- Visualisasi terbaik
st.subheader("📊 Visualisasi Hasil Terbaik (KMeans+PCA)")
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["KMeans_PCA_Label"], s=18)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"KMeans+PCA (K={best_k_pca}) — Silhouette={best_sil_pca:.3f}")
ax.grid(alpha=0.3)
st.pyplot(fig)

# --- Statistik rata-rata per cluster
st.subheader("📈 Statistik Rata-Rata Tiap Fitur per Cluster (KMeans+PCA)")
summary = df.groupby("KMeans_PCA_Label")[[
    "Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count", "Sleep Duration (hours)",
    "Stress Level", "Activity_ord", "Activity_Score", "Health_Index",
    "Recovery_Score", "Fatigue_Index", "Balance_Score"
]].mean().round(2)
st.dataframe(summary)

# --- Export CSV
st.subheader("💾 Download Hasil")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download hasil CSV", csv, "smartwatch_clusters_result.csv", "text/csv")

st.success("Selesai ✅ Pipeline, PCA, dan pemilihan model berjalan otomatis!")
