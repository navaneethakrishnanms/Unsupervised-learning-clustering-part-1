import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

st.title("🔍 Clustering App – KMeans | DBSCAN | Agglomerative")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview", df.head())

    # Drop NA
    df = df.dropna()

    # Drop ID column if exists
    if "CUST_ID" in df.columns:
        df.drop("CUST_ID", axis=1, inplace=True)

    # Select numeric features
    df_numeric = df.select_dtypes(include=['float64', 'int64']).dropna()

    # Apply Yeo-Johnson Power Transformation
    pt = PowerTransformer(method='yeo-johnson')
    df_transformed = pt.fit_transform(df_numeric)

    # Standard Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_transformed)

    # Robust Scaling
    robust_scaler = RobustScaler()
    df_scaled = robust_scaler.fit_transform(df_numeric)

    # PCA
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(df_scaled)

    # --- Clustering Models ---
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
    dbscan = DBSCAN(eps=2.0, min_samples=5)
    agg = AgglomerativeClustering(n_clusters=4, linkage='ward')

    kmeans_labels = kmeans.fit_predict(X_pca)
    dbscan_labels = dbscan.fit_predict(X_pca)
    agg_labels = agg.fit_predict(X_pca)

    # --- Silhouette Scores ---
    scores = {}
    if len(set(kmeans_labels)) > 1:
        scores["KMeans"] = silhouette_score(X_pca, kmeans_labels)
    if len(set(dbscan_labels)) > 1:
        scores["DBSCAN"] = silhouette_score(X_pca, dbscan_labels)
    if len(set(agg_labels)) > 1:
        scores["Agglomerative"] = silhouette_score(X_pca, agg_labels)

    st.write("### 📈 Silhouette Scores")
    st.json(scores)

    # --- Visualizations ---
    models = {
        "KMeans": kmeans_labels,
        "DBSCAN": dbscan_labels,
        "Agglomerative": agg_labels
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, labels) in zip(axes, models.items()):
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=10)
        ax.set_title(name)
    st.pyplot(fig)

    st.success("✅ Clustering completed successfully!")
