Overview
DeepGCCA-MVF (Deep Generalized Canonical Correlation Analysis for Multi-View Fusion) is a multi-view fusion clustering method designed for single-cell RNA sequencing (scRNA-seq) data.
The pipeline works as follows: first, three complementary views (PCA, UMAP, and PHATE) are generated from the preprocessed expression matrix. A corrected version of DeepGCCA then fuses these views into a common latent space with learnable view weights. Finally, KMeans clustering is applied to the fused representation, and clustering quality is evaluated using ARI and NMI.

Key contributions:
Three complementary views: PCA (linear, global variance), UMAP (manifold, local neighborhood), PHATE (diffusion potential, branching structure).
Corrected DeepGCCA: independent per-view projection matrices, pure cross-view covariance loss, learnable view weights, and an encoder-decoder structure with reconstruction loss.

Project Structure
pretreat_data.py — Data preprocessing: cell filtering, normalization, log transformation, highly variable gene selection, Z-score standardization.
Three_View.py — Multi-view generation (PCA / UMAP / PHATE) with confidence estimation.
DEEPGCCA2.py — Corrected DeepGCCA model: independent projection matrices, cross-view covariance loss, encoder-decoder, learnable weights.
get_ari_and_nmi.py — Clustering evaluation metrics (ARI / NMI).
main_DEEPGCCA.py — Main entry point: data loading → preprocessing → view generation → training → clustering → evaluation and saving.

Dependencies
Python 3.9.25
NumPy 1.26.4
PyTorch 2.5.1 (CPU)
scikit-learn 1.5.1
umap-learn 0.5.3
PHATE 2.0.0
Pandas 2.3.3
Matplotlib 3.9.2
pip install numpy==1.26.4 torch==2.5.1 scikit-learn==1.5.1 umap-learn==0.5.3 phate==2.0.0 pandas==2.3.3 matplotlib==3.9.2

Quick Start
1. Clone the Repository
git clone https://github.com/huaizhu-suns/DeepGCCA.git
cd DeepGCCA-MVF
2. Prepare Your Data
Place the following two CSV files in your data directory:
RNA_true_counts.csv — Gene expression matrix: rows are cells, columns are genes. The first column contains cell identifiers.
cell_groups.csv — Cell group labels: the second column contains the ground-truth cluster labels. The number of rows must match the expression matrix.
Before running, update DATA_ROOT in main_DEEPGCCA.py to point to your actual data directory.
3. Run
python main_DEEPGCCA.py
4. Output
Results are saved to ./sim_result_final/


Hyperparameters
All key hyperparameters are defined at the top of main_DEEPGCCA.py:
SEED = 66 — Global random seed for reproducibility.
TARGET_DIM = 20 — Target dimensionality for PCA / UMAP / PHATE views.
DGCCA_HIDDEN_DIM = 64 — Hidden layer size of the DeepGCCA encoder.
DGCCA_COMMON_DIM = 18 — Dimension of the common latent fusion space.
DGCCA_EPOCHS = 200 — Maximum training epochs.
DGCCA_LOSS_THRESHOLD = 0.1 — Early stopping loss threshold.
DGCCA_LR = 1e-3 — Initial learning rate for the Adam optimizer.
Method Details
Data Preprocessing
Five steps: (1) filter out cells with fewer than 200 expressed genes; (2) normalize each cell to a total expression of 10⁴; (3) apply ln(X+1) transformation; (4) when the number of genes exceeds 2000, select the top 2000 highly variable genes by variance; (5) apply Z-score standardization and remove genes with zero standard deviation.

Multi-View Generation
PCA View: Principal Component Analysis via SVD. Linear dimensionality reduction preserving global variance structure. Confidence is measured by the explained variance ratio.
UMAP View: Uniform Manifold Approximation and Projection. Based on fuzzy topology and cross-entropy optimization. Preserves local neighborhood relationships. Confidence combines global distance correlation and local k-NN preservation rate.
PHATE View: Potential of Heat-diffusion for Affinity-based Transition Embedding. Based on diffusion operators and MDS. Preserves branching structures and smooth transitions. Confidence is computed similarly via distance preservation metrics.

Clustering Evaluation
ARI (Adjusted Rand Index): Ranges from −1 to 1. Measures clustering similarity corrected for chance. Higher is better.
NMI (Normalized Mutual Information): Ranges from 0 to 1. Measures the shared information between predicted clusters and ground-truth labels. Higher is better.
