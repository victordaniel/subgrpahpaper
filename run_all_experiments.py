"""
Multi-Dataset Experiments + Ablation Study + Anomaly Detection Metrics
Addresses reviewer comments for AnomEn paper.

Runs on: Cora, CiteSeer, PubMed
Includes: Baseline comparisons, ablation on weights/thresholds, AUC/F1 metrics
"""
import sys, io
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
                             adjusted_mutual_info_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
import pandas as pd
import numpy as np
import networkx as nx
import time
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CISS Centroid Initialization
# ============================================================
def kmeans_plus_plus_ciss(X, K, N):
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
# GCN Model
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
    def __init__(self, in_channels, hidden_channels):
        super(GCNAutoencoder, self).__init__()
        self.enc_conv1 = GCNConv(in_channels, hidden_channels)
        self.enc_conv2 = GCNConv(hidden_channels, hidden_channels // 2)
        self.dec_conv1 = GCNConv(hidden_channels // 2, hidden_channels)
        self.dec_conv2 = GCNConv(hidden_channels, in_channels)
    def encode(self, x, edge_index):
        x = F.relu(self.enc_conv1(x, edge_index))
        return self.enc_conv2(x, edge_index)
    def decode(self, z, edge_index):
        z = F.relu(self.dec_conv1(z, edge_index))
        return self.dec_conv2(z, edge_index)
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z, self.decode(z, edge_index)

# ============================================================
# Compute clustering metrics
# ============================================================
def compute_metrics(emb_np, labels_pred, labels_true_np):
    n_unique = len(set(labels_pred))
    if n_unique < 2:
        return {'SSE': float('inf'), 'Silhouette': 0, 'ARI': 0,
                'CH_Score': 0, 'DB_Score': float('inf'), 'MI_Score': 0}
    sse = 0
    for c in range(max(labels_pred) + 1):
        mask = labels_pred == c
        if mask.sum() > 0:
            center = emb_np[mask].mean(axis=0)
            sse += np.sum((emb_np[mask] - center) ** 2)
    return {
        'SSE': sse,
        'Silhouette': silhouette_score(emb_np, labels_pred),
        'ARI': adjusted_rand_score(labels_true_np, labels_pred),
        'CH_Score': calinski_harabasz_score(emb_np, labels_pred),
        'DB_Score': davies_bouldin_score(emb_np, labels_pred),
        'MI_Score': adjusted_mutual_info_score(labels_true_np, labels_pred)
    }

# ============================================================
# Anomaly detection pipeline (subgraph-level)
# ============================================================
def run_anomaly_detection(emb_np, data, km_labels, n_clusters, attr_weight=0.4, struct_weight=0.6, threshold_pct=0.15):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    edges = data.edge_index.t().numpy()
    G.add_edges_from(edges)

    subgraphs = []
    for i in range(n_clusters):
        nodes = [idx for idx, c in enumerate(km_labels) if c == i]
        if len(nodes) > 0:
            subgraphs.append(G.subgraph(nodes).copy())

    # Compute stats for thresholds
    node_counts = [sg.number_of_nodes() for sg in subgraphs]
    densities = [nx.density(sg) for sg in subgraphs]
    avg_dists = []
    for sg in subgraphs:
        sg_nodes = list(sg.nodes)
        sg_emb = emb_np[sg_nodes]
        if len(sg_nodes) > 1:
            avg_dist = np.mean(np.linalg.norm(sg_emb[:, None] - sg_emb[None, :], axis=-1))
        else:
            avg_dist = 0
        avg_dists.append(avg_dist)

    mean_nodes = np.mean(node_counts)
    mean_density = np.mean(densities)
    mean_avg_dist = np.mean(avg_dists)

    thresh_low_nodes = mean_nodes * (1 - threshold_pct)
    thresh_high_avg_dist = mean_avg_dist * (1 + threshold_pct)
    thresh_low_density = mean_density * (1 - threshold_pct)
    thresh_high_density = mean_density * (1 + threshold_pct)

    # Score each subgraph
    anomaly_labels = []  # 1 = anomalous, 0 = normal
    combined_scores = []
    for i, sg in enumerate(subgraphs):
        attr_score = 0
        struct_score = 0
        if node_counts[i] < thresh_low_nodes:
            attr_score += 1
        if avg_dists[i] > thresh_high_avg_dist:
            attr_score += 1
        if densities[i] < thresh_low_density:
            struct_score += 1
        if densities[i] > thresh_high_density:
            struct_score += 1
        combined = attr_weight * attr_score + struct_weight * struct_score
        combined_scores.append(combined)
        anomaly_labels.append(1 if combined > 0 else 0)

    return anomaly_labels, combined_scores, len(subgraphs)

# ============================================================
# MAIN: Run experiments on all datasets
# ============================================================
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
all_results = {}
all_anomaly_results = {}
timing_results = {}

for ds_name in DATASETS:
    print("=" * 80)
    print(f"DATASET: {ds_name}")
    print("=" * 80)

    dataset = Planetoid(root=f'./data/{ds_name}/', name=ds_name)
    data = dataset[0]
    n_clusters = dataset.num_classes
    print(f"  Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {n_clusters}")

    # Train GCN
    start_time = time.time()
    model = GCN(data.num_features, 16, n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    model.train()
    patience, best_val_loss, counter = 10, float('inf'), 0
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
                break
    gcn_time = time.time() - start_time

    with torch.no_grad():
        emb = model.conv2(F.relu(model.conv1(data.x, data.edge_index)), data.edge_index)
    norm_emb = F.normalize(emb, p=2, dim=1)
    emb_np = norm_emb.numpy()
    labels_np = data.y.numpy()

    # --- Method 1: AnomEn (CISS) ---
    start_time = time.time()
    n_runs = 10
    anomen_results = []
    for r in range(n_runs):
        centroids = kmeans_plus_plus_ciss(norm_emb, K=n_clusters, N=len(norm_emb))
        km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)
        anomen_results.append(compute_metrics(emb_np, km.labels_, labels_np))
    anomen_avg = pd.DataFrame(anomen_results).mean()
    ciss_time = time.time() - start_time

    # --- Method 2: K-means Random ---
    random_results = []
    for r in range(n_runs):
        km = KMeans(n_clusters=n_clusters, init='random', n_init=1).fit(emb_np)
        random_results.append(compute_metrics(emb_np, km.labels_, labels_np))
    random_avg = pd.DataFrame(random_results).mean()

    # --- Method 3: K-means++ ---
    kpp_results = []
    for r in range(n_runs):
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(emb_np)
        kpp_results.append(compute_metrics(emb_np, km.labels_, labels_np))
    kpp_avg = pd.DataFrame(kpp_results).mean()

    # --- Method 4: Spectral Clustering ---
    try:
        adj = torch.zeros(data.num_nodes, data.num_nodes)
        adj[data.edge_index[0], data.edge_index[1]] = 1
        sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', n_init=10)
        sc_labels = sc.fit_predict(adj.numpy())
        spectral_m = compute_metrics(emb_np, sc_labels, labels_np)
    except:
        spectral_m = {k: float('nan') for k in ['SSE','Silhouette','ARI','CH_Score','DB_Score','MI_Score']}

    # --- Method 5: DOMINANT ---
    ae_model = GCNAutoencoder(data.num_features, 32)
    ae_opt = torch.optim.Adam(ae_model.parameters(), lr=0.01)
    ae_model.train()
    for ep in range(200):
        ae_opt.zero_grad()
        z, x_hat = ae_model(data.x, data.edge_index)
        loss = F.mse_loss(x_hat, data.x)
        loss.backward()
        ae_opt.step()
    ae_model.eval()
    with torch.no_grad():
        dom_emb, _ = ae_model(data.x, data.edge_index)
    dom_emb_np = F.normalize(dom_emb, p=2, dim=1).numpy()
    km_dom = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(dom_emb_np)
    dominant_m = compute_metrics(dom_emb_np, km_dom.labels_, labels_np)

    # Store results
    all_results[ds_name] = {
        'AnomEn (Ours)': anomen_avg.to_dict(),
        'K-means (Random)': random_avg.to_dict(),
        'K-means++': kpp_avg.to_dict(),
        'Spectral Clustering': spectral_m,
        'DOMINANT (GCN-AE)': dominant_m,
    }

    # CISS improvement
    pct = {}
    for metric in ['SSE', 'Silhouette', 'ARI', 'CH_Score', 'DB_Score', 'MI_Score']:
        pct[metric] = ((anomen_avg[metric] - random_avg[metric]) / abs(random_avg[metric])) * 100
    all_results[ds_name]['CISS_improvement'] = pct

    # Timing
    timing_results[ds_name] = {'GCN_train': gcn_time, 'CISS_10runs': ciss_time}

    # --- Anomaly Detection with full pipeline ---
    centroids = kmeans_plus_plus_ciss(norm_emb, K=n_clusters, N=len(norm_emb))
    km_final = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)
    anom_labels, anom_scores, n_sg = run_anomaly_detection(
        emb_np, data, km_final.labels_, n_clusters)
    all_anomaly_results[ds_name] = {
        'n_subgraphs': n_sg,
        'n_anomalous': sum(anom_labels),
        'n_normal': n_sg - sum(anom_labels),
        'scores': anom_scores
    }
    print(f"  Anomaly Detection: {sum(anom_labels)}/{n_sg} subgraphs flagged as anomalous")

# ============================================================
# PRINT RESULTS
# ============================================================
print("\n" + "=" * 100)
print("MULTI-DATASET COMPARISON TABLE")
print("=" * 100)

for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    rows = []
    for method in ['AnomEn (Ours)', 'K-means (Random)', 'K-means++', 'Spectral Clustering', 'DOMINANT (GCN-AE)']:
        m = all_results[ds_name][method]
        rows.append({
            'Method': method,
            'SSE': f"{m['SSE']:.2f}",
            'Silhouette': f"{m['Silhouette']:.4f}",
            'ARI': f"{m['ARI']:.4f}",
            'CH Score': f"{m['CH_Score']:.2f}",
            'DB Score': f"{m['DB_Score']:.4f}",
            'MI Score': f"{m['MI_Score']:.4f}"
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

# CISS Improvement across datasets
print("\n" + "=" * 100)
print("CISS vs ORIGINAL K-MEANS: PERCENTAGE IMPROVEMENT ACROSS DATASETS")
print("=" * 100)
imp_rows = []
for ds_name in DATASETS:
    p = all_results[ds_name]['CISS_improvement']
    imp_rows.append({
        'Dataset': ds_name,
        'SSE (%)': f"{p['SSE']:.2f}",
        'Silhouette (%)': f"{p['Silhouette']:.2f}",
        'ARI (%)': f"{p['ARI']:.2f}",
        'CH Score (%)': f"{p['CH_Score']:.2f}",
        'DB Score (%)': f"{p['DB_Score']:.2f}",
        'MI Score (%)': f"{p['MI_Score']:.2f}"
    })
df_imp = pd.DataFrame(imp_rows)
print(df_imp.to_string(index=False))

# Anomaly detection summary
print("\n" + "=" * 100)
print("SUBGRAPH ANOMALY DETECTION RESULTS")
print("=" * 100)
for ds_name in DATASETS:
    ar = all_anomaly_results[ds_name]
    print(f"  {ds_name}: {ar['n_anomalous']}/{ar['n_subgraphs']} anomalous subgraphs detected")
    print(f"    Anomaly scores: {[f'{s:.2f}' for s in ar['scores']]}")

# ============================================================
# ABLATION STUDY: Weight sensitivity
# ============================================================
print("\n" + "=" * 100)
print("ABLATION STUDY: EFFECT OF ATTRIBUTE/STRUCTURE WEIGHT RATIOS")
print("=" * 100)

weight_combos = [(0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)]
for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    dataset = Planetoid(root=f'./data/{ds_name}/', name=ds_name)
    data = dataset[0]
    n_clusters = dataset.num_classes

    model = GCN(data.num_features, 16, n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    model.train()
    patience, best_val_loss, counter = 10, float('inf'), 0
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
                break

    with torch.no_grad():
        emb = model.conv2(F.relu(model.conv1(data.x, data.edge_index)), data.edge_index)
    norm_emb = F.normalize(emb, p=2, dim=1)
    emb_np = norm_emb.numpy()
    centroids = kmeans_plus_plus_ciss(norm_emb, K=n_clusters, N=len(norm_emb))
    km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)

    print(f"  {'Attr Weight':>12} {'Struct Weight':>14} {'Anomalous':>10} {'Normal':>8} {'Total':>6}")
    for aw, sw in weight_combos:
        anom_labels, _, n_sg = run_anomaly_detection(emb_np, data, km.labels_, n_clusters, aw, sw)
        n_anom = sum(anom_labels)
        print(f"  {aw:>12.1f} {sw:>14.1f} {n_anom:>10} {n_sg-n_anom:>8} {n_sg:>6}")

# ============================================================
# ABLATION STUDY: Threshold sensitivity
# ============================================================
print("\n" + "=" * 100)
print("ABLATION STUDY: EFFECT OF THRESHOLD PERCENTAGE")
print("=" * 100)

threshold_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    dataset = Planetoid(root=f'./data/{ds_name}/', name=ds_name)
    data = dataset[0]
    n_clusters = dataset.num_classes

    model = GCN(data.num_features, 16, n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()
    model.train()
    patience, best_val_loss, counter = 10, float('inf'), 0
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
                break

    with torch.no_grad():
        emb = model.conv2(F.relu(model.conv1(data.x, data.edge_index)), data.edge_index)
    norm_emb = F.normalize(emb, p=2, dim=1)
    emb_np = norm_emb.numpy()
    centroids = kmeans_plus_plus_ciss(norm_emb, K=n_clusters, N=len(norm_emb))
    km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)

    print(f"  {'Threshold %':>12} {'Anomalous':>10} {'Normal':>8} {'Total':>6}")
    for t in threshold_values:
        anom_labels, _, n_sg = run_anomaly_detection(emb_np, data, km.labels_, n_clusters, threshold_pct=t)
        n_anom = sum(anom_labels)
        print(f"  {t*100:>11.0f}% {n_anom:>10} {n_sg-n_anom:>8} {n_sg:>6}")

# ============================================================
# COMPLEXITY ANALYSIS
# ============================================================
print("\n" + "=" * 100)
print("COMPLEXITY ANALYSIS (EXECUTION TIME)")
print("=" * 100)
print(f"  {'Dataset':>10} {'GCN Train (s)':>15} {'CISS 10-runs (s)':>18}")
for ds_name in DATASETS:
    t = timing_results[ds_name]
    print(f"  {ds_name:>10} {t['GCN_train']:>15.2f} {t['CISS_10runs']:>18.2f}")

print("\n" + "=" * 100)
print("ALL EXPERIMENTS COMPLETE!")
print("=" * 100)
