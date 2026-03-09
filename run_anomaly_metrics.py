"""
Anomaly Detection Metrics (AUC, F1, Precision, Recall) + Combined Score Threshold Ablation
Injects synthetic anomalies into graph datasets to create ground truth labels.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from sklearn.cluster import KMeans
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score)
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CISS
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
# GCN
# ============================================================
class GCN(torch.nn.Module):
    def __init__(self, in_ch, hid_ch, out_ch):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hid_ch)
        self.conv2 = GCNConv(hid_ch, out_ch)
    def forward(self, x, ei):
        x = F.dropout(F.relu(self.conv1(x, ei)), training=self.training)
        return F.log_softmax(self.conv2(x, ei), dim=1)

# ============================================================
# Anomaly detection pipeline
# ============================================================
def run_anomaly_pipeline(emb_np, data, km_labels, n_clusters, 
                         attr_w=0.4, struct_w=0.6, thresh_pct=0.15, score_thresh=0.0):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.t().numpy())

    subgraphs = []
    cluster_nodes = []
    for i in range(n_clusters):
        nodes = [idx for idx, c in enumerate(km_labels) if c == i]
        if len(nodes) > 0:
            subgraphs.append(G.subgraph(nodes).copy())
            cluster_nodes.append(nodes)

    node_counts = [sg.number_of_nodes() for sg in subgraphs]
    densities = [nx.density(sg) for sg in subgraphs]
    avg_dists = []
    for sg in subgraphs:
        ns = list(sg.nodes)
        e = emb_np[ns]
        avg_dists.append(np.mean(np.linalg.norm(e[:, None] - e[None, :], axis=-1)) if len(ns) > 1 else 0)

    mn, md, ma = np.mean(node_counts), np.mean(densities), np.mean(avg_dists)
    t_ln = mn * (1 - thresh_pct)
    t_ha = ma * (1 + thresh_pct)
    t_ld = md * (1 - thresh_pct)
    t_hd = md * (1 + thresh_pct)

    # Per-node anomaly predictions
    node_scores = np.zeros(data.num_nodes)
    node_preds = np.zeros(data.num_nodes, dtype=int)

    for i, sg in enumerate(subgraphs):
        attr_score = 0
        struct_score = 0
        if node_counts[i] < t_ln: attr_score += 1
        if avg_dists[i] > t_ha: attr_score += 1
        if densities[i] < t_ld: struct_score += 1
        if densities[i] > t_hd: struct_score += 1
        combined = attr_w * attr_score + struct_w * struct_score

        for node in cluster_nodes[i]:
            node_scores[node] = combined
            node_preds[node] = 1 if combined > score_thresh else 0

    return node_preds, node_scores

# ============================================================
# Inject synthetic anomalies
# ============================================================
def inject_anomalies(data, anomaly_ratio=0.05):
    """Inject structural anomalies by rewiring edges for a subset of nodes."""
    n_anomalies = int(data.num_nodes * anomaly_ratio)
    anomaly_indices = np.random.choice(data.num_nodes, n_anomalies, replace=False)
    
    ground_truth = np.zeros(data.num_nodes, dtype=int)
    ground_truth[anomaly_indices] = 1
    
    # Perturb features of anomalous nodes
    data_copy = data.clone()
    for idx in anomaly_indices:
        # Shuffle features randomly
        perm = torch.randperm(data_copy.x.shape[1])
        data_copy.x[idx] = data_copy.x[idx][perm]
        # Add random noise
        data_copy.x[idx] += torch.randn_like(data_copy.x[idx]) * 0.5
    
    return data_copy, ground_truth

# ============================================================
# MAIN
# ============================================================
DATASETS = ['Cora', 'CiteSeer', 'PubMed']
np.random.seed(42)
torch.manual_seed(42)

print("=" * 90)
print("ANOMALY DETECTION METRICS (AUC, F1, Precision, Recall)")
print("Using synthetic anomaly injection (5% of nodes)")
print("=" * 90)

all_metrics = {}

for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    dataset = Planetoid(root=f'./data/{ds_name}/', name=ds_name)
    data = dataset[0]
    n_clusters = dataset.num_classes

    # Inject anomalies
    data_anom, gt_labels = inject_anomalies(data, anomaly_ratio=0.05)

    # Train GCN
    model = GCN(data_anom.num_features, 16, n_clusters)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    crit = torch.nn.NLLLoss()
    model.train()
    patience, best, cnt = 10, float('inf'), 0
    for ep in range(200):
        opt.zero_grad()
        out = model(data_anom.x, data_anom.edge_index)
        loss = crit(out[data_anom.train_mask], data_anom.y[data_anom.train_mask])
        loss.backward(); opt.step()
        vl = crit(out[data_anom.val_mask], data_anom.y[data_anom.val_mask])
        if vl < best: best, cnt = vl, 0
        else:
            cnt += 1
            if cnt >= patience: break

    with torch.no_grad():
        emb = model.conv2(F.relu(model.conv1(data_anom.x, data_anom.edge_index)), data_anom.edge_index)
    emb_np = F.normalize(emb, p=2, dim=1).numpy()

    # Run AnomEn pipeline
    centroids = kmeans_plus_plus_ciss(F.normalize(emb, p=2, dim=1), K=n_clusters, N=len(emb))
    km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)
    preds, scores = run_anomaly_pipeline(emb_np, data_anom, km.labels_, n_clusters)

    # Compute metrics
    auc = roc_auc_score(gt_labels, scores)
    f1 = f1_score(gt_labels, preds)
    prec = precision_score(gt_labels, preds, zero_division=0)
    rec = recall_score(gt_labels, preds)
    acc = accuracy_score(gt_labels, preds)

    all_metrics[ds_name] = {'AUC': auc, 'F1': f1, 'Precision': prec, 'Recall': rec, 'Accuracy': acc}
    print(f"  AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, Accuracy: {acc:.4f}")

# Print LaTeX table
print("\n" + "=" * 90)
print("LATEX TABLE: Anomaly Detection Metrics")
print("=" * 90)
print(r"""
\begin{table}[ht]
\centering
\caption{Anomaly Detection Performance with Synthetic Anomalies (5\% injection rate)}
\label{tab:anomaly_metrics}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Dataset} & \textbf{AUC} & \textbf{F1} & \textbf{Precision} & \textbf{Recall} & \textbf{Accuracy} \\
\hline""")
for ds_name in DATASETS:
    m = all_metrics[ds_name]
    print(f"{ds_name} & {m['AUC']:.4f} & {m['F1']:.4f} & {m['Precision']:.4f} & {m['Recall']:.4f} & {m['Accuracy']:.4f} \\\\")
    print(r"\hline")
print(r"""\end{tabular}
\end{table}""")

# ============================================================
# COMBINED SCORE THRESHOLD ABLATION
# ============================================================
print("\n" + "=" * 90)
print("ABLATION: COMBINED SCORE THRESHOLD")
print("=" * 90)

score_thresholds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

for ds_name in DATASETS:
    print(f"\n--- {ds_name} ---")
    dataset = Planetoid(root=f'./data/{ds_name}/', name=ds_name)
    data = dataset[0]
    n_clusters = dataset.num_classes
    data_anom, gt_labels = inject_anomalies(data, anomaly_ratio=0.05)

    model = GCN(data_anom.num_features, 16, n_clusters)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    crit = torch.nn.NLLLoss()
    model.train()
    patience, best, cnt = 10, float('inf'), 0
    for ep in range(200):
        opt.zero_grad()
        out = model(data_anom.x, data_anom.edge_index)
        loss = crit(out[data_anom.train_mask], data_anom.y[data_anom.train_mask])
        loss.backward(); opt.step()
        vl = crit(out[data_anom.val_mask], data_anom.y[data_anom.val_mask])
        if vl < best: best, cnt = vl, 0
        else:
            cnt += 1
            if cnt >= patience: break

    with torch.no_grad():
        emb = model.conv2(F.relu(model.conv1(data_anom.x, data_anom.edge_index)), data_anom.edge_index)
    norm_emb = F.normalize(emb, p=2, dim=1)
    emb_np = norm_emb.numpy()
    centroids = kmeans_plus_plus_ciss(norm_emb, K=n_clusters, N=len(norm_emb))
    km = KMeans(n_clusters=n_clusters, init=centroids.numpy(), n_init=1).fit(emb_np)

    print(f"  {'Threshold':>10} {'AUC':>8} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}")
    for st in score_thresholds:
        preds, scores = run_anomaly_pipeline(emb_np, data_anom, km.labels_, n_clusters, score_thresh=st)
        try:
            auc = roc_auc_score(gt_labels, scores)
        except:
            auc = 0.0
        f1 = f1_score(gt_labels, preds, zero_division=0)
        prec = precision_score(gt_labels, preds, zero_division=0)
        rec = recall_score(gt_labels, preds, zero_division=0)
        acc = accuracy_score(gt_labels, preds)
        print(f"  {st:>10.1f} {auc:>8.4f} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {acc:>8.4f}")

print("\n" + "=" * 90)
print("ALL METRICS COMPLETE!")
print("=" * 90)
