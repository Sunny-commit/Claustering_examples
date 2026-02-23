# 🧠 Clustering Examples - Unsupervised Learning Techniques

A **comprehensive unsupervised learning portfolio** showcasing clustering algorithms including K-Means, hierarchical clustering, DBSCAN, and advanced techniques like Gaussian Mixture Models with complete visualization and evaluation.

## 🎯 Overview

This project demonstrates:
- ✅ K-Means clustering
- ✅ Hierarchical clustering (Agglomerative)
- ✅ DBSCAN (density-based)
- ✅ Gaussian Mixture Models (GMM)
- ✅ Silhouette analysis
- ✅ Elbow method for optimal k
- ✅ 2D/3D visualization
- ✅ Real-world dataset applications

## 🏗️ Architecture

### Clustering Pipeline
```
Data → Preprocessing → Feature Scaling → Clustering Algorithm → Evaluation → Visualization
                                              ├─ K-Means
                                              ├─ Hierarchical
                                              ├─ DBSCAN
                                              └─ GMM
```

### Tech Stack
| Component | Technology |
|-----------|-----------|
| **Core** | scikit-learn, SciPy |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Distance Metrics** | Euclidean, Manhattan, Cosine |
| **Evaluation** | Silhouette, Davies-Bouldin, Calinski-Harabasz |

## 📊 Dataset Preparation

### Feature Scaling (Critical for distance-based clustering)

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load dataset
data = pd.read_csv('customer_data.csv')
print(data.head())
print(data.describe())

# Remove missing values
data = data.dropna()

# Select numeric features
features = data[['Age', 'Income', 'Spending_Score', 'Recency']]

# Scale features (very important!)
# Without scaling, features with larger ranges dominate distance calculations
scaler = StandardScaler()  # Mean=0, Std=1
X_scaled = scaler.fit_transform(features)

print(f"Original shape: {features.shape}")
print(f"Scaled data mean: {X_scaled.mean(axis=0):.4f}")
print(f"Scaled data std: {X_scaled.std(axis=0):.4f}")
```

## 🔍 K-Means Clustering

### Algorithm & Implementation

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Step 1: Determine optimal number of clusters (Elbow Method)
inertias = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow Method Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Inertia plot
axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-Cluster Sum of Squares)')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Silhouette plot
axes[1].plot(k_range, silhouette_scores, 'g^-', linewidth=2)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score Analysis')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Optimal k is where elbow appears (k=3 or 4 typically)
optimal_k = 3
```

### K-Means Training & Analysis

```python
# Step 2: Train K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Step 3: Evaluate
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
db_score = davies_bouldin_score(X_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")  # -1 to 1, higher is better
print(f"Davies-Bouldin Index: {db_score:.3f}")    # Lower is better
print(f"Cluster sizes: {np.bincount(cluster_labels)}")

# Step 4: Extract results
clusters = pd.DataFrame({
    'cluster': cluster_labels,
    'Age': features['Age'],
    'Income': features['Income'],
    'Spending': features['Spending_Score']
})

# Cluster statistics
for i in range(optimal_k):
    cluster_data = clusters[clusters['cluster'] == i]
    print(f"\nCluster {i} (n={len(cluster_data)}):")
    print(f"  Avg Age: {cluster_data['Age'].mean():.1f}")
    print(f"  Avg Income: {cluster_data['Income'].mean():.0f}")
    print(f"  Avg Spending: {cluster_data['Spending'].mean():.1f}")
```

### K-Means Visualization

```python
# 2D Visualization (using PCA for dimensionality reduction)
from sklearn.decomposition import PCA

# Reduce to 2D for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot clusters
plt.figure(figsize=(10, 6))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for i in range(optimal_k):
    mask = cluster_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
               label=f'Cluster {i}', alpha=0.6, s=50)

# Plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
           marker='*', s=300, c='black', edgecolors='yellow', 
           linewidths=2, label='Centroids')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('K-Means Clustering Results (PCA Projection)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Silhouette plot
from matplotlib import cm

fig, ax = plt.subplots(figsize=(10, 6))
y_lower = 10

for i in range(optimal_k):
    cluster_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    cluster_silhouette_values = cluster_silhouette_values[cluster_labels == i]
    cluster_silhouette_values.sort()
    
    size_cluster_i = cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    y_lower = y_upper + 10

ax.set_xlabel("Silhouette Coefficient")
ax.set_ylabel("Cluster Label")
ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
          label=f'Average: {silhouette_avg:.3f}')
ax.set_title("Silhouette Plot for Each Cluster")
plt.legend()
plt.show()
```

## 🌲 Hierarchical Clustering

### Agglomerative Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Step 1: Compute linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')

# Step 2: Plot dendrogram
plt.figure(figsize=(14, 6))
dendrogram(linkage_matrix, leaf_rotation=90.0, leaf_font_size=8)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.axhline(y=10, color='red', linestyle='--', label='Cutting threshold')
plt.legend()
plt.tight_layout()
plt.show()

# Step 3: Fit hierarchical clustering
hierarchical = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'  # Other options: 'complete', 'average', 'single'
)
hier_labels = hierarchical.fit_predict(X_scaled)

# Evaluate
print(f"Hierarchical Silhouette Score: {silhouette_score(X_scaled, hier_labels):.3f}")
```

### Linkage Methods Comparison

```
Linkage Methods:
- Ward: Minimizes within-cluster variance (good default)
- Complete: Maximum distance between clusters
- Average: Average distance
- Single: Minimum distance (produces elongated clusters)

Use Ward for balanced, roughly spherical clusters
Use Complete for separated, compact clusters
Use Average as balanced compromise
Use Single for elongated, chain-like structures
```

## 🎯 DBSCAN (Density-Based)

### Epsilon & Min-Points Selection

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Step 1: Find optimal epsilon using k-distance graph
neighbors = NearestNeighbors(n_neighbors=5)  # MinPts = 5
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, -1], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(distances)
plt.xlabel('Data Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.title('K-distance Graph for Epsilon Selection')
plt.axhline(y=2.5, color='red', linestyle='--', label='Epsilon ≈ 2.5')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Step 2: Train DBSCAN with selected parameters
eps = 2.5
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)

# Step 3: Analyze results
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise}")

if n_clusters > 1:
    silhouette = silhouette_score(X_scaled, dbscan_labels)
    print(f"Silhouette Score: {silhouette:.3f}")
```

### DBSCAN Visualization

```python
# Visualize DBSCAN results
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
colors = plt.cm.Spectral(np.linspace(0, 1, n_clusters))

# Plot clusters
for i in range(n_clusters):
    mask = dbscan_labels == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=f'Cluster {i}', s=50, alpha=0.6)

# Plot noise points
noise_mask = dbscan_labels == -1
plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
           c='black', marker='x', s=100, 
           label='Noise', linewidths=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('DBSCAN Clustering Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 🎲 Gaussian Mixture Models (GMM)

### Probabilistic Clustering

```python
from sklearn.mixture import GaussianMixture

# Step 1: Determine optimal components using BIC/AIC
bic_scores = []
aic_scores = []
n_components_range = range(1, 11)

for n_comp in n_components_range:
    gmm = GaussianMixture(n_components=n_comp, random_state=42)
    gmm.fit(X_scaled)
    bic_scores.append(gmm.bic(X_scaled))
    aic_scores.append(gmm.aic(X_scaled))

# Plot model selection
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(n_components_range, bic_scores, 'o-', label='BIC', linewidth=2)
ax.plot(n_components_range, aic_scores, 's-', label='AIC', linewidth=2)
ax.set_xlabel('Number of Components')
ax.set_ylabel('Score')
ax.set_title('Model Selection: BIC and AIC')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# Step 2: Fit GMM with optimal components
optimal_components = 3  # Based on BIC/AIC
gmm = GaussianMixture(n_components=optimal_components, covariance_type='full',
                     random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_proba = gmm.predict_proba(X_scaled)  # Soft assignments

# Step 3: Results
print(f"Silhouette Score: {silhouette_score(X_scaled, gmm_labels):.3f}")

# Get soft assignments (probabilities for each cluster)
for i, (label, probs) in enumerate(zip(gmm_labels[:5], gmm_proba[:5])):
    print(f"\nSample {i}: Cluster {label}")
    for j, prob in enumerate(probs):
        print(f"  P(Cluster {j}) = {prob:.3f}")
```

## 📊 Algorithm Comparison

### Performance Metrics

| Algorithm | Shape | Scalability | Densities | Parameters |
|-----------|-------|-------------|-----------|-----------|
| **K-Means** | Spherical | O(nkd) | Even | k |
| **Hierarchical** | Any | O(n²) | Even | Linkage |
| **DBSCAN** | Arbitrary | O(n log n) | Varying | eps, min_pts |
| **GMM** | Elliptical | O(nkd) | Even | k, covariance |

### When to Use Each

```python
# Use K-Means for:
# - Fast clustering of large datasets
# - Roughly spherical clusters
# - Known number of clusters
# Example: Customer segmentation (3-5 clear segments)

# Use Hierarchical for:
# - Dendrograms (understanding cluster structure)
# - Variable cluster sizes
# - Small-medium datasets
# Example: Gene clustering in biology

# Use DBSCAN for:
# - Arbitrary cluster shapes
# - Noise detection
# - Clusters of varying density
# Example: Spatial clustering, outlier detection

# Use GMM for:
# - Soft assignments (probabilities)
# - Overlapping clusters
# - Probabilistic model needed
# Example: Mixture of populations
```

## 💡 Interview Questions

### Q: How do you choose k in K-Means?
```
Answer - Three methods:
1. Elbow Method: Plot inertia vs k, look for bend
2. Silhouette Score: Higher = better separation
3. Domain knowledge: Business context determines clusters

Also consider: computational cost vs accuracy trade-off
```

### Q: Difference between K-Means and DBSCAN?
```
K-Means:
- Partitions all points into k clusters
- Assumes spherical, similarly-sized clusters
- Every point assigned (no noise)

DBSCAN:
- Finds dense regions
- Can detect noise points
- Works with arbitrary shapes
- No pre-specified k
```

## 🌟 Portfolio Value

✅ Unsupervised learning mastery
✅ Multiple algorithms & comparisons
✅ Proper evaluation metrics
✅ Real-world interpretable results
✅ Visualization expertise
✅ Model selection techniques
✅ Production-ready approach

## 📄 License

MIT License - Educational Use

---

**Next Steps**:
1. Add semi-supervised learning
2. Implement deep clustering
3. Add time-series clustering
4. Streaming k-means for online data
5. Interactive clustering visualization
