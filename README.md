# 📊 Clustering Examples - Unsupervised Learning

A comprehensive **collection of clustering algorithms** demonstrating K-Means, hierarchical clustering, DBSCAN, and Gaussian mixture models with practical examples and visualizations.

## 🎯 Overview

This project covers:
- ✅ K-Means clustering
- ✅ Hierarchical clustering (agglomerative)
- ✅ DBSCAN (density-based)
- ✅ Gaussian Mixture Models
- ✅ Elbow method & silhouette analysis
- ✅ Cluster visualization

## 🎯 K-Means Clustering

```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class KMeansClusterer:
    """K-Means clustering implementation"""
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        self.scaler = StandardScaler()
    
    def fit_predict(self, data):
        """Fit and predict clusters"""
        # Normalize data
        data_scaled = self.scaler.fit_transform(data)
        
        # Fit K-Means
        clusters = self.model.fit_predict(data_scaled)
        
        return clusters, self.model.cluster_centers_, self.model.inertia_
    
    def elbow_method(self, data, k_range=range(1, 11)):
        """Find optimal k using elbow method"""
        inertias = []
        
        data_scaled = self.scaler.fit_transform(data)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data_scaled)
            inertias.append(kmeans.inertia_)
        
        # Plot elbow
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')
        plt.grid()
        plt.show()
        
        return inertias
    
    def silhouette_analysis(self, data):
        """Silhouette score for cluster quality"""
        from sklearn.metrics import silhouette_score
        
        data_scaled = self.scaler.fit_transform(data)
        
        scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data_scaled)
            score = silhouette_score(data_scaled, labels)
            scores.append(score)
        
        best_k = k_range[np.argmax(scores)]
        
        return best_k, scores
```

## 🌳 Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

class HierarchicalClusterer:
    """Hierarchical (agglomerative) clustering"""
    
    def __init__(self, linkage_method='ward'):
        """
        linkage_method: 'ward', 'complete', 'average', 'single'
        """
        self.linkage_method = linkage_method
        self.linkage_matrix = None
    
    def fit(self, data):
        """Perform hierarchical clustering"""
        self.scaler = StandardScaler()
        data_scaled = self.scaler.fit_transform(data)
        
        # Compute linkage matrix
        self.linkage_matrix = linkage(data_scaled, method=self.linkage_method)
        
        return self.linkage_matrix
    
    def dendrogram_plot(self, labels=None, figsize=(14, 7)):
        """Plot dendrogram"""
        plt.figure(figsize=figsize)
        dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90
        )
        plt.title(f'Dendrogram ({self.linkage_method} linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()
    
    def get_clusters(self, n_clusters):
        """Get cluster assignments"""
        clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        return clusters - 1  # 0-indexed
    
    def compare_linkage_methods(self, data):
        """Compare different linkage methods"""
        methods = ['ward', 'complete', 'average', 'single']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, method in enumerate(methods):
            linkage_matrix = linkage(data, method=method)
            ax = axes[idx]
            dendrogram(linkage_matrix, ax=ax)
            ax.set_title(f'{method.capitalize()} Linkage')
        
        plt.tight_layout()
        plt.show()
```

## 🔴 DBSCAN - Density-Based Clustering

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

class DBSCANClusterer:
    """Density-based clustering"""
    
    def __init__(self, eps=0.3, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
    
    def fit_predict(self, data):
        """Fit and predict clusters"""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        clusters = self.model.fit_predict(data_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        n_noise = list(clusters).count(-1)
        
        return {
            'labels': clusters,
            'n_clusters': n_clusters,
            'n_noise_points': n_noise
        }
    
    def find_optimal_eps(self, data, k=5):
        """Find eps using k-distance graph"""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Compute k-distances
        nbrs = NearestNeighbors(n_neighbors=k).fit(data_scaled)
        distances, indices = nbrs.kneighbors(data_scaled)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Plot k-distance graph
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.ylabel(f'{k}-distance')
        plt.title('K-distance Graph (look for elbow)')
        plt.grid()
        plt.show()
        
        return distances
```

## 🎯 Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

class GMMClusterer:
    """Gaussian Mixture Models"""
    
    def __init__(self, n_components=3):
        self.n_components = n_components
        self.model = GaussianMixture(n_components=n_components, random_state=42)
    
    def fit_predict(self, data):
        """Fit GMM and predict"""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Fit model
        labels = self.model.fit_predict(data_scaled)
        
        # Get probabilities
        probabilities = self.model.predict_proba(data_scaled)
        
        return labels, probabilities
    
    def bic_comparison(self, data, n_range=range(1, 11)):
        """Compare BIC scores"""
        scores = []
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        for n in n_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(data_scaled)
            scores.append(gmm.bic(data_scaled))
        
        best_n = n_range[np.argmin(scores)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(n_range, scores, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('BIC')
        plt.title('BIC Score vs Number of Components')
        plt.axvline(best_n, color='r', linestyle='--')
        plt.show()
        
        return best_n, scores
```

## 📊 Clustering Metrics

```python
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score
)

class ClusteringMetrics:
    """Evaluate clustering quality"""
    
    @staticmethod
    def evaluate(data, labels):
        """Calculate clustering metrics"""
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Internal metrics
        silhouette = silhouette_score(data_scaled, labels)
        davies_bouldin = davies_bouldin_score(data_scaled, labels)
        calinski = calinski_harabasz_score(data_scaled, labels)
        
        return {
            'Silhouette': silhouette,  # Higher is better (-1 to 1)
            'Davies-Bouldin': davies_bouldin,  # Lower is better
            'Calinski-Harabasz': calinski  # Higher is better
        }
    
    @staticmethod
    def compare_labels(true_labels, predicted_labels):
        """Compare with ground truth"""
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        
        return {
            'Adjusted Rand Index': ari,  # Higher is better (-1 to 1)
            'Normalized Mutual Info': nmi  # Higher is better (0 to 1)
        }
```

## 💡 Interview Talking Points

**Q: When use K-Means vs Hierarchical vs DBSCAN?**
```
Answer:
- K-Means: Fast, assumes spherical clusters, need to know k
- Hierarchical: Dendrogram view, captures hierarchy
- DBSCAN: Arbitrary shapes, handles outliers, eps-sensitive
```

**Q: How find optimal number of clusters?**
```
Answer:
- Elbow method (K-Means)
- Silhouette score
- BIC/AIC (GMM)
- Domain knowledge
```

## 🌟 Portfolio Value

✅ Multiple clustering algorithms
✅ Cluster evaluation metrics
✅ Visualization techniques
✅ Hyperparameter selection
✅ Algorithm comparison
✅ Unsupervised learning

---

**Technologies**: Scikit-learn, NumPy, Pandas, Matplotlib, SciPy

