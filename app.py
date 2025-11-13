# SMARTWATCH HEALTH CLUSTERING — Streamlit (KMeans + PCA Only, with rich visualization)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# --- Setup dasar
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
st.set_page_config(page_title="Smartwatch Health Clustering", layout="wide")

st.title("⌚ Smartwatch Health Clustering (KMeans + PCA)")
st.markdown("""
Pipeline otomatis: **Impute → Scaling → Yeo-Johnson → Log Transform → PCA → KMeans → Visualisasi**  
Upload data smartwatch-mu dan lihat hasil klasternya secara interaktif.
""")

# --- Upload File
uploaded_file = st.file_uploader("📁 Upload file CSV (smartwatch_health.csv)", type=["csv"])
if uploaded_file is None:
    st.info("Silakan upload file CSV terlebih dahulu.")
    st.stop()

df = pd.read_csv(uploaded_file)
st.subheader("Preview Data")
st.dataframe(df.head())

# --- Pilih fitur utama
features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
            "Sleep Duration (hours)", "Stress Level"]

# Tambah kolom Activity jika ada
if "Activity Level" in df.columns:
    mapping = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
    df["Activity_ord"] = df["Activity Level"].map(mapping).fillna(1)
else:
    df["Activity_ord"] = 1
features.append("Activity_ord")

# Pastikan numeric
df[features] = df[features].apply(pd.to_numeric, errors="coerce")

# --- Buat fitur turunan
df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
df["Health_Index"] = (df["Blood Oxygen Level (%)"] / df["Heart Rate (BPM)"]) * 100
df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)

# --- Preprocessing
st.subheader("⚙️ Preprocessing Data")
imputer = SimpleImputer(strategy='median')
df[features] = imputer.fit_transform(df[features])

scaler = RobustScaler()
X_scaled = scaler.fit_transform(df[features])

pt = PowerTransformer(method='yeo-johnson')
X_power = pt.fit_transform(X_scaled)

X_log = np.log1p(np.abs(X_power))
X_log = np.nan_to_num(X_log, nan=0.0)

# --- PCA
pca = PCA(n_components=3, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_log)
st.success("✅ PCA berhasil dilakukan (3 komponen utama).")

# --- KMeans Clustering
st.subheader("🤖 KMeans Clustering")
k = st.slider("Pilih jumlah cluster (K)", 2, 10, 3)
kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
labels = kmeans.fit_predict(X_pca)
df["Cluster"] = labels
sil_score = silhouette_score(X_pca, labels)
st.write(f"**Silhouette Score:** {sil_score:.3f}")

# --- PCA Variance
exp_var = pca.explained_variance_ratio_ * 100
st.write(f"Komponen PCA menjelaskan varian total: {exp_var.sum():.2f}%")
st.bar_chart(pd.DataFrame(exp_var, index=["PC1", "PC2", "PC3"], columns=["Explained Variance (%)"]))

# --- Scatter Plot 2D
st.subheader("📊 Visualisasi PCA 2D")
fig2d, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, s=25)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title(f"PCA 2D Scatter (K={k})")
plt.colorbar(sc, ax=ax, label="Cluster")
st.pyplot(fig2d)

# --- Scatter Plot 3D
st.subheader("🌈 Visualisasi PCA 3D")
from mpl_toolkits.mplot3d import Axes3D
fig3d = plt.figure(figsize=(8, 6))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=labels, s=20)
ax3d.set_xlabel("PC1")
ax3d.set_ylabel("PC2")
ax3d.set_zlabel("PC3")
ax3d.set_title(f"PCA 3D Scatter (K={k})")
st.pyplot(fig3d)

# --- Rata-rata per Cluster
st.subheader("📈 Rata-Rata Tiap Fitur per Cluster")
summary = df.groupby("Cluster")[features + [
    "Activity_Score", "Health_Index", "Recovery_Score",
    "Fatigue_Index", "Balance_Score"
]].mean().round(2)
st.dataframe(summary)

# --- Bar Chart per Fitur
st.subheader("📊 Perbandingan Fitur per Cluster (Bar Chart)")
for col in ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
            "Sleep Duration (hours)", "Stress Level"]:
    fig_bar, ax = plt.subplots()
    summary[col].plot(kind="bar", ax=ax)
    ax.set_title(f"{col} per Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel(col)
    st.pyplot(fig_bar)

# --- Heatmap cluster summary
st.subheader("🔥 Heatmap Ringkasan Cluster")
fig_heat, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(summary.values, cmap='coolwarm', aspect='auto')
ax.set_xticks(np.arange(len(summary.columns)))
ax.set_yticks(np.arange(len(summary.index)))
ax.set_xticklabels(summary.columns, rotation=45, ha="right")
ax.set_yticklabels(summary.index)
plt.colorbar(im, ax=ax, label="Mean Value")
ax.set_title("Heatmap Mean Fitur per Cluster")
st.pyplot(fig_heat)

# --- Distribusi cluster (pie chart)
st.subheader("🍩 Distribusi Cluster")
fig_pie, ax = plt.subplots()
counts = df["Cluster"].value_counts().sort_index()
ax.pie(counts, labels=[f"Cluster {i}" for i in counts.index], autopct="%1.1f%%")
ax.set_title("Distribusi Jumlah Data per Cluster")
st.pyplot(fig_pie)

# --- Download hasil
st.subheader("💾 Download Hasil Clustering")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV Hasil", csv, "smartwatch_clusters_kmeans.csv", "text/csv")

st.success("🎉 Selesai — Clustering dan visualisasi berhasil dibuat!")
