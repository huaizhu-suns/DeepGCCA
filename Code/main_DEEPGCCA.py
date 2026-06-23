import os
import pickle
import warnings
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
from get_ari_and_nmi import compute_clustering_metrics
from pretreat_data import pretreat_data
from DEEPGCCA2 import train_corrected_deepgcca
from Three_View import (
    generate_pca_view,
    generate_umap_view,
    generate_phate_view
)
warnings.filterwarnings("ignore")

# ===================== 超参数配置 ======================
SEED = 48
TARGET_DIM = 20
DGCCA_HIDDEN_DIM = 64
DGCCA_COMMON_DIM = 18
DGCCA_EPOCHS = 200
DGCCA_LOSS_THRESHOLD = 0.1
DGCCA_LR = 1e-3
SAVE_DIR = "./sim_result_final"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== 数据集路径（Mouse 数据） ======================
DATA_ROOT = r"C:\Users\17436\Desktop\DEEPGCCA012\data_temp"
H5AD_FILE = "Exp_Seurat.h5ad"
LABEL_COLUMN = 'clusters_fig2_final'
h5ad_path = os.path.join(DATA_ROOT, H5AD_FILE)

# ===================== 固定随机种子 ======================
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"训练设备：{device}")

# ===================== 步骤1：读取 Mouse 数据 ======================
print("【1/5】读取 Mouse 单细胞数据（Exp_Seurat.h5ad）...")
adata = ad.read_h5ad(h5ad_path)

X_raw = adata.X
if hasattr(X_raw, 'toarray'):
    X_raw = X_raw.toarray()
X_raw = X_raw.astype(np.float32)

if LABEL_COLUMN not in adata.obs.columns:
    raise KeyError(f"标签列 '{LABEL_COLUMN}' 不存在。可用列: {list(adata.obs.columns)}")
true_labels_raw = adata.obs[LABEL_COLUMN].values

mask = ~pd.isna(true_labels_raw)
X_raw = X_raw[mask]
true_labels_raw = true_labels_raw[mask]

uniq = np.unique(true_labels_raw)
mapping = {lab: i for i, lab in enumerate(uniq)}
true_labs = np.array([mapping[lab] for lab in true_labels_raw])

n_clusters = len(uniq)
print(f"  表达矩阵维度：{X_raw.shape}（细胞数×基因数）")
print(f"  真实分组数：{n_clusters}，标签维度：{true_labs.shape}")
print(f"  标签分布：{np.bincount(true_labs)}")

# ===================== 步骤2：数据预处理 ======================
print("【2/5】数据预处理...")
datasets = {"Simulated_scRNA": pretreat_data(X_raw)}
X = datasets["Simulated_scRNA"]
print(f"  预处理后特征矩阵维度：{X.shape}")

# ===================== 步骤3：生成3个视图（PCA、UMAP、PHATE） ======================
print("【3/5】生成3个多视图特征（PCA / UMAP / PHATE）...")
view_pca   = generate_pca_view(X, TARGET_DIM, 42)     # ← 修正：去掉 confidence_pca
view_umap  = generate_umap_view(X, TARGET_DIM, 42)    # ← 修正：去掉 confidence_umap
view_phate = generate_phate_view(X, TARGET_DIM, 42)   # ← 修正：去掉 confidence_phate

views_np = {
    "pca": view_pca,
    "umap": view_umap,
    "phate": view_phate
}
for vname, v in views_np.items():
    print(f"  {vname}视图维度：{v.shape}（细胞数×{TARGET_DIM}）")

views_tensor = {k: torch.tensor(v, dtype=torch.float32).to(device) for k, v in views_np.items()}

# ===================== 步骤4：DeepGCCA 多视图融合 ======================
print("【4/5】训练DeepGCCA...")

view_dims = [views_np[name].shape[1] for name in views_np.keys()]
print(f"  视图维度: {dict(zip(views_np.keys(), view_dims))}")

views_list = [torch.tensor(views_np[name], dtype=torch.float32).to(device)
              for name in views_np.keys()]

fused_representation, view_weights, deepgcca_model = train_corrected_deepgcca(
    views_tensor=views_list,
    view_dims=view_dims,
    hidden_dim=DGCCA_HIDDEN_DIM,
    common_dim=DGCCA_COMMON_DIM,
    epochs=DGCCA_EPOCHS,
    loss_threshold=DGCCA_LOSS_THRESHOLD,
    lr=DGCCA_LR
)

print(f"DeepGCCA融合完成！")
print(f"  - 融合表示形状: {fused_representation.shape}")
print(f"  - 最终视图权重: {dict(zip(views_np.keys(), view_weights))}")

# ===================== 步骤5：DeepGCCA 融合特征 KMeans 聚类与保存 ======================
print("【5/5】DeepGCCA融合特征KMeans聚类...")

kmeans_kwargs = {
    "n_clusters": n_clusters,
    "n_init": "auto",
    "random_state": SEED,
    "max_iter": 300
}

kmeans_deepgcca = KMeans(**kmeans_kwargs)
deepgcca_pred = kmeans_deepgcca.fit_predict(fused_representation)
deepgcca_ari, deepgcca_nmi = compute_clustering_metrics(true_labs, deepgcca_pred)
print(f"  DeepGCCA+KMeans：ARI={deepgcca_ari:.4f}，NMI={deepgcca_nmi:.4f}")

# 保存结果
np.save(os.path.join(SAVE_DIR, "mouse_deepgcca_label.npy"), deepgcca_pred)
np.save(os.path.join(SAVE_DIR, "mouse_deepgcca_fused.npy"), fused_representation)
np.save(os.path.join(SAVE_DIR, "mouse_view_weights.npy"), view_weights)
print(f"  结果已保存至 {SAVE_DIR}")

# ===================== 最终结果 ======================
print("\n========== DeepGCCA 最终聚类结果 ==========")
print(f"DeepGCCA + KMeans       : ARI={deepgcca_ari:.4f}, NMI={deepgcca_nmi:.4f}")

print("\n完成。")
