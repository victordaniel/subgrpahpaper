"""
Script to reproduce Table 2 from the paper:
SSE values of the CISS method and the original K-means algorithm (10 runs)
Also produces Tables 3-8 and Table 1 (differences) and Table 8 (% improvement).

Based on the original code from subgrpahpaper-main.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_mutual_info_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 1: Load the Cora dataset
# ============================================================
print("Loading Cora dataset...")
dataset = Planetoid(root='./data/Cora/', name='Cora')
data = dataset[0]
labels = data.y
num_classes = dataset.num_classes
print(f"Dataset loaded: {data.num_nodes} nodes, {data.num_features} features, {num_classes} classes")

# ============================================================
# Step 2: CISS (Centroid Initialization by Selective Sampling)
# ============================================================
def kmeans_plus_plus_ciss(X, K, N):
    """
    CISS: Centroid Initialization by Selective Sampling.
    Selects first centroid as closest to feature mean, then iteratively
    removes N/K nearest points before selecting next centroid.
    """
    centroids = []
    X_remaining = X.clone()
    N_remaining = N
    
    for k in range(K):
        if k == 0:
            feature_means = torch.mean(X_remaining, dim=0)
            distances = torch.sqrt(torch.sum((X_remaining - feature_means)**2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X_remaining[closest_index])
        else:
            dist_to_centroid = torch.sqrt(torch.sum((X_remaining - centroids[k-1])**2, dim=1))
            sorted_indices = torch.argsort(dist_to_centroid)
            delete_indices = sorted_indices[:N_remaining // K]
            remaining_indices = list(set(range(N_remaining)) - set(delete_indices.tolist()))
            X_remaining = X_remaining[remaining_indices]
            N_remaining = len(X_remaining)
            feature_means = torch.mean(X_remaining, dim=0)
            distances = torch.sqrt(torch.sum((X_remaining - feature_means)**2, dim=1))
            closest_index = torch.argmin(distances)
            centroids.append(X_remaining[closest_index])
    return torch.stack(centroids)

# ============================================================
# Step 3: Define GCN Model
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ============================================================
# Step 4: Train GCN with Early Stopping
# ============================================================
print("\nTraining GCN model...")
model = GCN(data.num_features, 16, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.NLLLoss()
model.train()

patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Training stopped after {epoch} epochs (early stopping).")
            break

# ============================================================
# Step 5: Extract and Normalize Embeddings
# ============================================================
print("Extracting node embeddings...")
with torch.no_grad():
    embeddings = model.conv2(
        F.relu(model.conv1(data.x, data.edge_index)),
        data.edge_index
    )
normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
print(f"Embeddings shape: {normalized_embeddings.shape}")

# ============================================================
# Step 6: Run Clustering Comparison (10 runs)
# ============================================================
n_clusters = 7

def run_clustering(normalized_embeddings, centroids, true_labels, n_clusters=7):
    """Run both CISS K-means and Original K-means, return all metrics."""
    # CISS K-means
    proposed_kmeans = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(normalized_embeddings.numpy())
    proposed_sse = proposed_kmeans.inertia_
    proposed_sil = silhouette_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_ari = adjusted_rand_score(true_labels.numpy(), proposed_kmeans.labels_)
    proposed_ch = calinski_harabasz_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_db = davies_bouldin_score(normalized_embeddings.numpy(), proposed_kmeans.labels_)
    proposed_mi = adjusted_mutual_info_score(true_labels.numpy(), proposed_kmeans.labels_)

    # Original K-means (random init)
    original_kmeans = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(normalized_embeddings.numpy())
    original_sse = original_kmeans.inertia_
    original_sil = silhouette_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_ari = adjusted_rand_score(true_labels.numpy(), original_kmeans.labels_)
    original_ch = calinski_harabasz_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_db = davies_bouldin_score(normalized_embeddings.numpy(), original_kmeans.labels_)
    original_mi = adjusted_mutual_info_score(true_labels.numpy(), original_kmeans.labels_)

    return {
        'CISS_SSE': proposed_sse, 'Original_SSE': original_sse,
        'CISS_Silhouette': proposed_sil, 'Original_Silhouette': original_sil,
        'CISS_ARI': proposed_ari, 'Original_ARI': original_ari,
        'CISS_CH': proposed_ch, 'Original_CH': original_ch,
        'CISS_DB': proposed_db, 'Original_DB': original_db,
        'CISS_MI': proposed_mi, 'Original_MI': original_mi,
    }

n_runs = 10
print(f"\nRunning {n_runs} clustering experiments...")
results_list = []

for run in range(n_runs):
    centroids = kmeans_plus_plus_ciss(normalized_embeddings, K=7, N=len(normalized_embeddings))
    result = run_clustering(normalized_embeddings, centroids, data.y, n_clusters)
    results_list.append(result)
    print(f"  Run {run+1}/10 complete")

results_df = pd.DataFrame(results_list)

# ============================================================
# OUTPUT: Table 2 - SSE Values
# ============================================================
print("\n" + "="*70)
print("TABLE 2: SSE values of the CISS method and the original K-means algorithm")
print("="*70)
sse_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_SSE'].values,
    'Original K-means': results_df['Original_SSE'].values
})
print(sse_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 3 - Silhouette Scores
# ============================================================
print("\n" + "="*70)
print("TABLE 3: Silhouette scores of the CISS method and the original K-means algorithm")
print("="*70)
sil_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_Silhouette'].values,
    'Original K-means': results_df['Original_Silhouette'].values
})
print(sil_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 4 - ARI Scores
# ============================================================
print("\n" + "="*70)
print("TABLE 4: ARI scores of the CISS method and the original K-means algorithm")
print("="*70)
ari_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_ARI'].values,
    'Original K-means': results_df['Original_ARI'].values
})
print(ari_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 5 - CH Scores
# ============================================================
print("\n" + "="*70)
print("TABLE 5: CH scores of the CISS method and the original K-means algorithm")
print("="*70)
ch_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_CH'].values,
    'Original K-means': results_df['Original_CH'].values
})
print(ch_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 6 - DB Scores
# ============================================================
print("\n" + "="*70)
print("TABLE 6: DB scores of the CISS method and the original K-means algorithm")
print("="*70)
db_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_DB'].values,
    'Original K-means': results_df['Original_DB'].values
})
print(db_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 7 - MI Scores
# ============================================================
print("\n" + "="*70)
print("TABLE 7: MI scores of the CISS method and the original K-means algorithm")
print("="*70)
mi_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'CISS': results_df['CISS_MI'].values,
    'Original K-means': results_df['Original_MI'].values
})
print(mi_table.to_string(index=False, float_format='%.4f'))

# ============================================================
# OUTPUT: Table 1 - Differences (CISS - Original)
# ============================================================
print("\n" + "="*70)
print("TABLE 1: Differences in performance metrics (CISS - Original K-means)")
print("="*70)
diff_table = pd.DataFrame({
    'Run': range(1, n_runs+1),
    'SSE': results_df['CISS_SSE'].values - results_df['Original_SSE'].values,
    'Silhouette': results_df['CISS_Silhouette'].values - results_df['Original_Silhouette'].values,
    'ARI': results_df['CISS_ARI'].values - results_df['Original_ARI'].values,
    'CH Score': results_df['CISS_CH'].values - results_df['Original_CH'].values,
    'DB Score': results_df['CISS_DB'].values - results_df['Original_DB'].values,
    'MI Score': results_df['CISS_MI'].values - results_df['Original_MI'].values,
})
print(diff_table.to_string(index=False, float_format='%.6f'))

# ============================================================
# OUTPUT: Table 8 - Percentage Improvement
# ============================================================
print("\n" + "="*70)
print("TABLE 8: Performance of Anomaly Detection Method on Cora Dataset")
print("="*70)
mean_results = results_df.mean()
metrics = {
    'SSE': ((mean_results['CISS_SSE'] - mean_results['Original_SSE']) / mean_results['Original_SSE']) * 100,
    'Silhouette Score': ((mean_results['CISS_Silhouette'] - mean_results['Original_Silhouette']) / mean_results['Original_Silhouette']) * 100,
    'Adjusted Rand Index (ARI)': ((mean_results['CISS_ARI'] - mean_results['Original_ARI']) / mean_results['Original_ARI']) * 100,
    'Calinski-Harabasz Score': ((mean_results['CISS_CH'] - mean_results['Original_CH']) / mean_results['Original_CH']) * 100,
    'Davies-Bouldin Score': ((mean_results['CISS_DB'] - mean_results['Original_DB']) / mean_results['Original_DB']) * 100,
    'Adjusted MI Score': ((mean_results['CISS_MI'] - mean_results['Original_MI']) / mean_results['Original_MI']) * 100,
}

for metric_name, improvement in metrics.items():
    print(f"  {metric_name:30s} : {improvement:+.2f}%")

print("\n" + "="*70)
print("All tables generated successfully!")
print("="*70)
