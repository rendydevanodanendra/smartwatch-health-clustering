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
            return None, None, None, None, None

    # 2. Definisi Fitur Awal
    features = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
                "Sleep Duration (hours)", "Stress Level"]

    # 3. Encode Activity Level
    if "Activity Level" in df.columns:
        activity_map = {"Sedentary": 0, "Active": 1, "Highly Active": 2}
        df["Activity_ord"] = df["Activity Level"].map(activity_map).fillna(1)
        features.append("Activity_ord")
    elif "Activity_ord" in df.columns:
         features.append("Activity_ord")

    # 4. Konversi ke Numerik
    for col in features:
        if col in df.columns and df[col].dtype == 'object':
             df[col] = df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5. Imputasi Awal
    imputer = SimpleImputer(strategy='median')
    valid_features = [f for f in features if f in df.columns]
    df[valid_features] = imputer.fit_transform(df[valid_features])

    # 6. Fitur Turunan
    if all(col in df.columns for col in ["Step Count", "Sleep Duration (hours)", "Blood Oxygen Level (%)", "Heart Rate (BPM)", "Stress Level"]):
        df["Activity_Score"] = df["Step Count"] / (df["Sleep Duration (hours)"] + 0.1)
        df["Health_Index"] = (df["Blood Oxygen Level (%)"] / (df["Heart Rate (BPM)"] + 1e-9)) * 100
        df["Recovery_Score"] = (df["Sleep Duration (hours)"] * df["Blood Oxygen Level (%)"]) / (df["Stress Level"] + 1)
        df["Fatigue_Index"] = (df["Heart Rate (BPM)"] * (df["Stress Level"] + 1)) / (df["Sleep Duration (hours)"] + 0.1)
        df["Balance_Score"] = df["Recovery_Score"] / (df["Fatigue_Index"] + 1)
        final_features = valid_features + ["Activity_Score", "Health_Index", "Recovery_Score", "Fatigue_Index", "Balance_Score"]
    else:
        final_features = valid_features

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df[final_features] = imputer.fit_transform(df[final_features])

    # 7. Scaling & Transformasi
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df[final_features])

    try:
        pt = PowerTransformer(method='yeo-johnson')
        X_power = pt.fit_transform(X_scaled)
    except:
        X_power = X_scaled

    X_for_pca = X_power

    # 8. PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_for_pca)
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    return df, final_features, X_scaled, X_power, X_pca

# --- SIDEBAR ---
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV (opsional)", type=["csv"])

with st.spinner("🔄 Memuat data..."):
    df, features, X_scaled, X_power, X_pca = load_and_preprocess_data(uploaded_file)

if df is None:
    st.warning("⚠️ File tidak ditemukan. Silakan upload file CSV.")
    st.stop()

st.sidebar.success("✅ Data siap!")
with st.sidebar.expander("ℹ️ Info Data"):
    st.write(f"Baris: {df.shape[0]}")
    st.write(f"Fitur: {len(features)}")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["📊 Ringkasan", "⚙️ Eksperimen", "💡 Interpretasi"])

# TAB 1
with tab1:
    st.subheader("Data (5 Baris Awal)")
    st.dataframe(df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Statistik Deskriptif**")
        st.dataframe(df[features].describe())
    with col2:
        st.write("**Korelasi Fitur**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[features].corr(), cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# TAB 2
with tab2:
    st.header("🛠️ Konfigurasi Eksperimen")
    c1, c2 = st.columns(2)
    with c1:
        model_name = st.selectbox("Algoritma", ["K-Means (PCA)", "GMM (PCA)", "Spectral (PCA) - LAMBAT!"])
    with c2:
        k_range = st.slider("Rentang K", 2, 10, (2, 5))

    # Opsi Sampling untuk Spectral
    use_sampling = False
    if "Spectral" in model_name and len(df) > 2000:
        st.warning(f"⚠️ Data Anda {len(df)} baris. Spectral Clustering akan sangat lambat atau crash.")
        use_sampling = st.checkbox("Gunakan Sampling (misal: 1000 data acak) agar lebih cepat?", value=True)

    if st.button("🚀 Jalankan Eksperimen"):
        results = []
        best_score = -1
        best_k = -1
        best_labels = None
        
        # Logika Sampling
        if use_sampling and "Spectral" in model_name:
            sample_size = min(1000, len(df))
            # Sample indeks, bukan datanya langsung agar nanti bisa di-merge balik jika perlu (agak tricky)
            # Untuk simplifikasi dashboard ini, kita jalankan eksperimen HANYA pada sampel
            indices = np.random.choice(len(df), sample_size, replace=False)
            X_input_full = X_pca if "PCA" in model_name else X_power
            X_input = X_input_full[indices]
            st.info(f"ℹ️ Menjalankan Spectral pada {sample_size} sampel data acak.")
        else:
             X_input = X_pca if "PCA" in model_name else X_power

        prog_bar = st.progress(0)
        
        k_vals = range(k_range[0], k_range[1] + 1)
        for i, k in enumerate(k_vals):
            try:
                if "K-Means" in model_name:
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(X_input)
                elif "GMM" in model_name:
                    model = GaussianMixture(n_components=k, random_state=42)
                    labels = model.fit_predict(X_input)
                elif "Spectral" in model_name:
                    # Jika tidak sampling dan data besar, skip paksa untuk mencegah crash server
                    if not use_sampling and len(df) > 5000:
                         st.error(f"K={k} dilewati: Data terlalu besar untuk Spectral tanpa sampling.")
                         continue
                    model = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
                    labels = model.fit_predict(X_input)

                sil = silhouette_score(X_input, labels)
                results.append({"K": k, "Silhouette Score": sil})

                if sil > best_score:
                    best_score = sil
                    best_k = k
                    # Jika sampling, kita tidak bisa langsung pakai labels untuk seluruh data df
                    # Jadi kita simpan None dulu jika sampling, nanti dihandle
                    best_labels = labels if not use_sampling else None 

            except Exception as e:
                st.error(f"Gagal untuk K={k}: {e}")

            prog_bar.progress((i + 1) / len(k_vals))

        prog_bar.empty()

        if not results:
            st.error("❌ Tidak ada hasil eksperimen yang berhasil dijalankan.")
        else:
            results_df = pd.DataFrame(results)
            st.session_state['results_df'] = results_df
            st.session_state['best_k'] = best_k
            st.session_state['best_score'] = best_score
            st.session_state['best_model'] = model_name
            
            # Handle label untuk visualisasi
            if best_labels is not None:
                st.session_state['best_labels'] = best_labels
                st.session_state['is_sampled'] = False
            elif use_sampling and best_k != -1:
                # Jika sampling, kita harus memprediksi label untuk SEMUA data
                # Spectral tidak punya method .predict(), jadi kita pakai K-Nearest Neighbors
                # untuk mengaproksimasi label sisa data berdasarkan sampel yang sudah dilabeli.
                from sklearn.neighbors import KNeighborsClassifier
                knn = KNeighborsClassifier(n_neighbors=1)
                knn.fit(X_input, labels) # Fit pada sampel
                full_labels = knn.predict(X_pca if "PCA" in model_name else X_power) # Prediksi semua
                st.session_state['best_labels'] = full_labels
                st.session_state['is_sampled'] = True

    # Tampilkan Hasil
    if 'results_df' in st.session_state and not st.session_state['results_df'].empty:
        st.subheader("📈 Hasil Eksperimen")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.dataframe(st.session_state['results_df'].style.highlight_max(axis=0, color='lightgreen', subset=['Silhouette Score']))
            st.success(f"🏆 K Terbaik: **{st.session_state['best_k']}** (Score: {st.session_state['best_score']:.3f})")
            if st.session_state.get('is_sampled', False):
                st.warning("⚠️ Hasil berdasarkan sampling. Label untuk seluruh data adalah aproksimasi.")
        with c2:
            fig, ax = plt.subplots()
            sns.lineplot(data=st.session_state['results_df'], x="K", y="Silhouette Score", marker="o", ax=ax)
            ax.set_title("Silhouette Score vs K")
            st.pyplot(fig)

        # Visualisasi
        if 'best_labels' in st.session_state:
            df['Cluster'] = st.session_state['best_labels']
            st.subheader(f"Visualisasi Cluster Terbaik (K={st.session_state['best_k']})")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df, palette='bright', s=50, ax=ax)
            ax.set_title(f"Proyeksi PCA - {st.session_state['best_model']}")
            st.pyplot(fig)

# TAB 3
with tab3:
    if 'Cluster' in df.columns:
        st.header("💡 Interpretasi")
        st.write("Rata-rata Fitur per Cluster:")
        st.dataframe(df.groupby('Cluster')[features].mean().style.background_gradient(cmap='Blues'))
        
        feat = st.selectbox("Pilih fitur untuk boxplot:", features)
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.boxplot(x='Cluster', y=feat, data=df, palette='viridis', ax=ax)
        st.pyplot(fig)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil CSV", csv, "clustering_results.csv", "text/csv")
    else:
        st.info("Jalankan eksperimen dulu di Tab 2.")
