# app.py
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

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

st.set_page_config(page_title="Smartwatch Health Clustering", layout="wide")

st.title("⌚ Smartwatch Health Clustering (Colab-like pipeline)")
st.markdown("Upload CSV, pipeline akan dijalankan sama persis seperti di Colab (impute → robust scale → yeo-johnson → log → PCA → compare models).")

# Show versions (helpful to ensure parity)
st.sidebar.subheader("Environment info")
st.sidebar.write(f"scikit-learn: {sklearn.__version__}")
st.sidebar.write(f"pandas: {pd.__version__}")
st.sidebar.write(f"numpy: {np.__version__}")

uploaded_file = st.file_uploader("Upload file CSV (smartwatch_health.csv)", type=["csv"])

if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

# Read file
df = pd.read_csv(uploaded_file)
st.subheader("Preview data")
st.dataframe(df.head())

# --- Features setup (sama seperti di Colab)
features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
            "Sleep Duration (hours)", "Stress Level"]

# Encode activity level (jika ada kolom Activity Level)
if "Activity Level" in df.columns:
    activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
else:
    # jika tidak ada, buat default 1 (active) agar pipeline tetap jalan
    df["Activity_ord"] = 1

features.append("Activity_ord")

# Ensure numeric columns
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

# Create derived features (keberadaan tidak mengubah PCA input, tapi kita buat karena Colab membuatnya)
# (Colab membuat fitur turunan setelah konversi numeric)
df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
df["Health_Index"] = (df["Blood Oxygen Level (%)"] / df["Heart Rate (BPM)"]) * 100
df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)

st.write("Data siap digunakan. Kolom fitur:")
st.write(features)

# Imputation (median) on features list
df[features] = df[features].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])

# --- Scaling & Transformasi (identik dengan Colab)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[features])

pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X_scaled)

# log-transform sesuai Colab:
X_log = np.log1p(np.abs(X_power))
X_log = np.nan_to_num(X_log, nan=0.0)

# PCA (3 komponen)
pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_log)

st.success("Preprocessing selesai (median impute → RobustScaler → Yeo-Johnson → log1p(abs) → PCA).")

# --- RUN MODELS and compare (automated like your Colab)
st.subheader("🔍 Perbandingan Model (otomatis cari best K menurut silhouette)")

results_rows = []

# KMeans + PCA (loop K)
sil_kmeans_pca = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = kmeans.fit_predict(X_pca)
    try:
        sil = silhouette_score(X_pca, labels)
    except Exception:
        sil = np.nan
    sil_kmeans_pca[k] = sil
    st.write(f"[KMeans+PCA] K={k}, Silhouette={sil:.3f}")

best_k_pca = max(sil_kmeans_pca, key=sil_kmeans_pca.get)
best_model_kmeans = KMeans(n_clusters=best_k_pca, random_state=RANDOM_STATE, n_init=20).fit(X_pca)
df["KMeans_PCA_Label"] = best_model_kmeans.labels_

# KMeans RAW (tanpa PCA): menggunakan X_log
sil_kmeans_raw = {}
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X_log)
    try:
        sil = silhouette_score(X_log, labels)
    except Exception:
        sil = np.nan
    sil_kmeans_raw[k] = sil
    st.write(f"[KMeans RAW] K={k}, Silhouette={sil:.3f}")

best_k_raw = max(sil_kmeans_raw, key=sil_kmeans_raw.get)
df["KMeans_Raw_Label"] = KMeans(n_clusters=best_k_raw, random_state=RANDOM_STATE, n_init=20).fit_predict(X_log)

# GMM (pada X_pca)
sil_gmm = {}
for k in range(2, 10):
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE)
    labels = gmm.fit_predict(X_pca)
    try:
        sil = silhouette_score(X_pca, labels)
    except Exception:
        sil = np.nan
    sil_gmm[k] = sil
    st.write(f"[GMM] K={k}, Silhouette={sil:.3f}")

best_k_gmm = max(sil_gmm, key=sil_gmm.get)
gmm_best = GaussianMixture(n_components=best_k_gmm, random_state=RANDOM_STATE).fit(X_pca)
df["GMM_Label"] = gmm_best.predict(X_pca)

# Spectral Clustering (pada X_pca)
sil_spec = {}
for k in range(2, 10):
    spec = SpectralClustering(n_clusters=k, affinity='rbf', assign_labels='kmeans', random_state=RANDOM_STATE)
    # fit_predict kadang memperingatkan; tangani error
    try:
        labels = spec.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
    except Exception:
        labels = np.zeros(X_pca.shape[0], dtype=int)
        sil = np.nan
    sil_spec[k] = sil
    st.write(f"[Spectral] K={k}, Silhouette={sil:.3f}")

best_k_spec = max(sil_spec, key=sil_spec.get)
try:
    df["Spectral_Label"] = SpectralClustering(n_clusters=best_k_spec, affinity='rbf',
                                              assign_labels='kmeans', random_state=RANDOM_STATE).fit_predict(X_pca)
except Exception:
    # fallback jika spectral gagal
    df["Spectral_Label"] = df["KMeans_PCA_Label"]

# Ringkasan hasil
results = pd.DataFrame({
    "Model": ["KMeans+PCA", "KMeans RAW", "GMM", "Spectral"],
    "Best K": [best_k_pca, best_k_raw, best_k_gmm, best_k_spec],
    "Silhouette": [
        sil_kmeans_pca[best_k_pca],
        sil_kmeans_raw[best_k_raw],
        sil_gmm[best_k_gmm],
        sil_spec[best_k_spec]
    ]
}).sort_values(by="Silhouette", ascending=False).reset_index(drop=True)

st.subheader("=== 🔥 HASIL PERBANDINGAN AKHIR ===")
st.dataframe(results)

# Visualisasi hasil terbaik (KMeans+PCA)
st.subheader("📊 Visualisasi hasil terbaik (KMeans+PCA) — 2D PCA (PC1 vs PC2)")
fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=df["KMeans_PCA_Label"], s=18)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"KMeans+PCA (K={best_k_pca}) — Silhouette={sil_kmeans_pca[best_k_pca]:.3f}")
ax.grid(alpha=0.3)
st.pyplot(fig)

# Tabel ringkasan per cluster (KMeans+PCA)
st.subheader("📈 Statistik rata-rata tiap fitur per cluster (KMeans+PCA)")
summary = df.groupby("KMeans_PCA_Label")[
    ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count", "Sleep Duration (hours)",
     "Stress Level", "Activity_ord", "Activity_Score", "Health_Index",
     "Recovery_Score", "Fatigue_Index", "Balance_Score"]
].mean().round(2)
st.dataframe(summary)

# Export CSV hasil
st.subheader("💾 Download hasil")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download hasil CSV", data=csv, file_name="smartwatch_clusters_comparison_v_streamlit.csv", mime="text/csv")

st.success("Selesai — pipeline dan pemilihan model dilakukan sama persis seperti di Colab.")
