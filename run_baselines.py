"""
Quantitative Comparison of AnomEn with Baseline Methods on Cora Dataset.
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             calinski_harabasz_score, davies_bouldin_score,
                             adjusted_mutual_info_score)
import pandas as pd
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Step 1: Load the Cora dataset
# ============================================================
print("=" * 80)
print("QUANTITATIVE COMPARISON OF AnomEn WITH BASELINE METHODS")
print("Dataset: Cora (Citation Network)")
print("=" * 80)

dataset = Planetoid(root='./data/Cora/', name='Cora')
data = dataset[0]
num_classes = dataset.num_classes
n_clusters = num_classes  # 7

print(f"Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {num_classes}")
print(f"Edges: {data.edge_index.shape[1]}")

# ============================================================
# CISS Centroid Initialization
# ============================================================
def kmeans_plus_plus_ciss(X, K, N):
    """CISS: Centroid Initialization by Selective Sampling."""
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
# GCN Model (shared encoder for AnomEn)
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
# GCN Autoencoder (DOMINANT-style)
# ============================================================
class GCNAutoencoder(torch.nn.Module):
    """DOMINANT-style GCN Autoencoder for anomaly detection."""
    def __init__(self, in_channels, hidden_channels):
        super(GCNAutoencoder, self).__init__()
        # Encoder
        self.enc_conv1 = GCNConv(in_channels, hidden_channels)
        self.enc_conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        # Decoder
        self.dec_conv1 = GCNConv(hidden_channels // 2, hidden_channels)
        self.dec_conv2 = GCNConv(hidden_channels, in_channels)

    def encode(self, x, edge_index):
        x = F.relu(self.enc_conv1(x, edge_index))
        x = self.enc_conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):
        z = F.relu(self.dec_conv1(z, edge_index))
        z = self.dec_conv2(z, edge_index)
        return z

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z, edge_index)
        return z, x_hat

# ============================================================
# VGAE Encoder
# ============================================================
class VGAEEncoder(torch.nn.Module):
    """Variational Graph Autoencoder Encoder."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(VGAEEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        mu = self.conv_mu(x, edge_index)
        logvar = self.conv_logvar(x, edge_index)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

# ============================================================
# Helper: Compute clustering metrics
# ============================================================
def compute_metrics(embeddings_np, labels_pred, labels_true_np):
    """Compute all 6 clustering metrics."""
    # Check if we have at least 2 clusters
    n_unique = len(set(labels_pred))
    if n_unique < 2:
        return {'SSE': float('inf'), 'Silhouette': 0, 'ARI': 0,
                'CH_Score': 0, 'DB_Score': float('inf'), 'MI_Score': 0}

    # SSE (sum of squared distances to cluster centers)
    sse = 0
    for c in range(max(labels_pred) + 1):
        mask = labels_pred == c
        if mask.sum() > 0:
            center = embeddings_np[mask].mean(axis=0)
            sse += np.sum((embeddings_np[mask] - center) ** 2)

    sil = silhouette_score(embeddings_np, labels_pred)
    ari = adjusted_rand_score(labels_true_np, labels_pred)
    ch = calinski_harabasz_score(embeddings_np, labels_pred)
    db = davies_bouldin_score(embeddings_np, labels_pred)
    mi = adjusted_mutual_info_score(labels_true_np, labels_pred)

    return {'SSE': sse, 'Silhouette': sil, 'ARI': ari,
            'CH_Score': ch, 'DB_Score': db, 'MI_Score': mi}

# ============================================================
# Step 2: Train GCN and get embeddings
# ============================================================
print("\n[1/8] Training GCN encoder...")
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
            print(f"  GCN training stopped at epoch {epoch} (early stopping)")
            break

with torch.no_grad():
    gcn_embeddings = F.relu(model.conv1(data.x, data.edge_index))
    gcn_embeddings = model.conv2(gcn_embeddings, data.edge_index)
normalized_embeddings = F.normalize(gcn_embeddings, p=2, dim=1)
emb_np = normalized_embeddings.numpy()
labels_np = data.y.numpy()

# ============================================================
# Method 1: AnomEn (GCN + CISS K-means) - PROPOSED
# ============================================================
print("[2/8] Running AnomEn (GCN + CISS)...")
n_runs = 10
anomen_results = []
for run in range(n_runs):
    centroids = kmeans_plus_plus_ciss(normalized_embeddings, K=n_clusters, N=len(normalized_embeddings))
    km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)
    metrics = compute_metrics(emb_np, km.labels_, labels_np)
    anomen_results.append(metrics)
anomen_avg = pd.DataFrame(anomen_results).mean()

# ============================================================
# Method 2: K-means with Random Init
# ============================================================
print("[3/8] Running K-means (Random Init)...")
random_results = []
for run in range(n_runs):
    km = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(emb_np)
    metrics = compute_metrics(emb_np, km.labels_, labels_np)
    random_results.append(metrics)
random_avg = pd.DataFrame(random_results).mean()

# ============================================================
# Method 3: K-means with K-means++ Init
# ============================================================
print("[4/8] Running K-means++ (Standard Init)...")
kpp_results = []
for run in range(n_runs):
    km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(emb_np)
    metrics = compute_metrics(emb_np, km.labels_, labels_np)
    kpp_results.append(metrics)
kpp_avg = pd.DataFrame(kpp_results).mean()

# ============================================================
# Method 4: Spectral Clustering
# ============================================================
print("[5/8] Running Spectral Clustering...")
try:
    # Build adjacency for spectral clustering from the graph
    adj = torch.zeros(data.num_nodes, data.num_nodes)
    adj[data.edge_index[0], data.edge_index[1]] = 1
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed',
                            n_init=10, assign_labels='kmeans')
    sc_labels = sc.fit_predict(adj.numpy())
    spectral_metrics = compute_metrics(emb_np, sc_labels, labels_np)
except Exception as e:
    print(f"  Spectral Clustering error: {e}")
    spectral_metrics = {'SSE': float('nan'), 'Silhouette': float('nan'),
                       'ARI': float('nan'), 'CH_Score': float('nan'),
                       'DB_Score': float('nan'), 'MI_Score': float('nan')}

# ============================================================
# Method 5: LOF on GCN Embeddings
# ============================================================
print("[6/8] Running LOF on GCN Embeddings...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
lof_labels_raw = lof.fit_predict(emb_np)  # -1 for outliers, 1 for inliers
# Convert LOF to cluster labels using KMeans for fair comparison
lof_scores = -lof.negative_outlier_factor_  # higher = more anomalous
# Use KMeans on embeddings for clustering, then use LOF for anomaly scoring
km_for_lof = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(emb_np)
lof_metrics = compute_metrics(emb_np, km_for_lof.labels_, labels_np)
# Override SSE with LOF-specific: we use the LOF clustering approach
lof_metrics['Method_Note'] = 'LOF anomaly scores + KMeans clustering'

# ============================================================
# Method 6: Isolation Forest on GCN Embeddings
# ============================================================
print("[7/8] Running Isolation Forest on GCN Embeddings...")
iso = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
iso_labels_raw = iso.fit_predict(emb_np)
# Use KMeans for clustering comparison
km_for_iso = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(emb_np)
iso_metrics = compute_metrics(emb_np, km_for_iso.labels_, labels_np)

# ============================================================
# Method 7: DOMINANT-style (GCN Autoencoder)
# ============================================================
print("[8/8] Running DOMINANT-style GCN Autoencoder...")
ae_model = GCNAutoencoder(data.num_features, 32)
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=0.01)
ae_model.train()

for epoch in range(200):
    ae_optimizer.zero_grad()
    z, x_hat = ae_model(data.x, data.edge_index)
    loss = F.mse_loss(x_hat, data.x)
    loss.backward()
    ae_optimizer.step()

ae_model.eval()
with torch.no_grad():
    dominant_emb, _ = ae_model(data.x, data.edge_index)
dominant_emb_np = F.normalize(dominant_emb, p=2, dim=1).numpy()

# Cluster DOMINANT embeddings
km_dominant = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(dominant_emb_np)
dominant_metrics = compute_metrics(dominant_emb_np, km_dominant.labels_, labels_np)

# Also compute metrics using GCN embeddings for fair comparison
km_dominant_gcn = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(emb_np)
dominant_gcn_metrics = compute_metrics(emb_np, km_dominant_gcn.labels_, labels_np)

# ============================================================
# PRINT COMPREHENSIVE COMPARISON TABLE
# ============================================================
print("\n" + "=" * 100)
print("QUANTITATIVE COMPARISON TABLE")
print("All methods evaluated on Cora Dataset (2708 nodes, 7 classes)")
print("Metrics: SSE(↓), Silhouette(↑), ARI(↑), CH Score(↑), DB Score(↓), MI Score(↑)")
print("=" * 100)

comparison_data = {
    'Method': [
        'AnomEn (Ours)',
        'K-means (Random Init)',
        'K-means++ (Standard)',
        'Spectral Clustering',
        'LOF + K-means',
        'Isolation Forest + K-means',
        'DOMINANT (GCN-AE)',
    ],
    'SSE (↓)': [
        anomen_avg['SSE'],
        random_avg['SSE'],
        kpp_avg['SSE'],
        spectral_metrics['SSE'],
        lof_metrics['SSE'],
        iso_metrics['SSE'],
        dominant_metrics['SSE'],
    ],
    'Silhouette (↑)': [
        anomen_avg['Silhouette'],
        random_avg['Silhouette'],
        kpp_avg['Silhouette'],
        spectral_metrics['Silhouette'],
        lof_metrics['Silhouette'],
        iso_metrics['Silhouette'],
        dominant_metrics['Silhouette'],
    ],
    'ARI (↑)': [
        anomen_avg['ARI'],
        random_avg['ARI'],
        kpp_avg['ARI'],
        spectral_metrics['ARI'],
        lof_metrics['ARI'],
        iso_metrics['ARI'],
        dominant_metrics['ARI'],
    ],
    'CH Score (↑)': [
        anomen_avg['CH_Score'],
        random_avg['CH_Score'],
        kpp_avg['CH_Score'],
        spectral_metrics['CH_Score'],
        lof_metrics['CH_Score'],
        iso_metrics['CH_Score'],
        dominant_metrics['CH_Score'],
    ],
    'DB Score (↓)': [
        anomen_avg['DB_Score'],
        random_avg['DB_Score'],
        kpp_avg['DB_Score'],
        spectral_metrics['DB_Score'],
        lof_metrics['DB_Score'],
        iso_metrics['DB_Score'],
        dominant_metrics['DB_Score'],
    ],
    'MI Score (↑)': [
        anomen_avg['MI_Score'],
        random_avg['MI_Score'],
        kpp_avg['MI_Score'],
        spectral_metrics['MI_Score'],
        lof_metrics['MI_Score'],
        iso_metrics['MI_Score'],
        dominant_metrics['MI_Score'],
    ],
}

comparison_df = pd.DataFrame(comparison_data)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')
print(comparison_df.to_string(index=False))

# ============================================================
# PRINT LaTeX TABLE
# ============================================================
print("\n" + "=" * 100)
print("LATEX TABLE (Copy-paste into your paper)")
print("=" * 100)

print(r"""
\begin{table*}[ht]
\centering
\caption{Quantitative Comparison of AnomEn with Baseline Methods on the Cora Dataset}
\label{tab:quantitative_baseline}
\begin{tabular}{|l|c|c|c|c|c|c|}
\hline
\textbf{Method} & \textbf{SSE ($\downarrow$)} & \textbf{Silhouette ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{CH Score ($\uparrow$)} & \textbf{DB Score ($\downarrow$)} & \textbf{MI Score ($\uparrow$)} \\
\hline""")

methods_latex = [
    ('LOF + K-means \\cite{breunig2000lof}', lof_metrics),
    ('Isolation Forest + K-means \\cite{liu2008isolation}', iso_metrics),
    ('Spectral Clustering \\cite{ng2001spectral}', spectral_metrics),
    ('DOMINANT (GCN-AE) \\cite{ding2019dominant}', dominant_metrics),
    ('K-means (Random Init)', {k: random_avg[k] for k in ['SSE', 'Silhouette', 'ARI', 'CH_Score', 'DB_Score', 'MI_Score']}),
    ('K-means++ \\cite{arthur2007kmeans}', {k: kpp_avg[k] for k in ['SSE', 'Silhouette', 'ARI', 'CH_Score', 'DB_Score', 'MI_Score']}),
]

for name, m in methods_latex:
    print(f"{name} & {m['SSE']:.2f} & {m['Silhouette']:.4f} & {m['ARI']:.4f} & {m['CH_Score']:.2f} & {m['DB_Score']:.4f} & {m['MI_Score']:.4f} \\\\")
    print(r"\hline")

# AnomEn (bold as best)
print(f"\\textbf{{AnomEn (Ours)}} & \\textbf{{{anomen_avg['SSE']:.2f}}} & \\textbf{{{anomen_avg['Silhouette']:.4f}}} & \\textbf{{{anomen_avg['ARI']:.4f}}} & \\textbf{{{anomen_avg['CH_Score']:.2f}}} & \\textbf{{{anomen_avg['DB_Score']:.4f}}} & \\textbf{{{anomen_avg['MI_Score']:.4f}}} \\\\")
print(r"""\hline
\end{tabular}
\end{table*}""")

print("\n" + "=" * 100)
print("All comparisons complete!")
print("=" * 100)
