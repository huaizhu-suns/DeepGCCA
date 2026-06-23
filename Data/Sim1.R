# 0. 环境准备
rm(list = ls())
gc()
library(scMultiSim)

# 1. 加载GRN参数
data(GRN_params_100)

# 2. 生成模拟数据（严格用你提供的核心参数）
results <- sim_true_counts(list(
  GRN = GRN_params_100,
  tree = Phyla3(),
  num.cells = 1000,
  num.cif = 20,
  discrete.cif = FALSE,
  cif.sigma = 0.05,
  num.genes = 1800
))

# 3. 处理RNA矩阵
rna_mat <- as.matrix(results$counts)
rna_mat <- t(rna_mat)
cell_names <- paste0("cell", 1:nrow(rna_mat))
gene_names <- paste0("gene", 1:ncol(rna_mat))
rownames(rna_mat) <- cell_names
colnames(rna_mat) <- gene_names

# 4. 用cell_meta的pop列生成分组（你的输出里已存在pop列）
cell_group <- data.frame(
  Cell = cell_names,
  Group = results$cell_meta$pop
)

# 5. 保存文件
outdir <- "C:/Users/17436/Desktop/DEEPGCCA012/500 110 C"
dir.create(outdir, showWarnings = FALSE)

# 保存RNA矩阵
rna_out <- cbind(Cell = rownames(rna_mat), rna_mat)
write.table(
  rna_out,
  file = file.path(outdir, "RNA_true_counts.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

# 保存分组
write.table(
  cell_group,
  file = file.path(outdir, "cell_groups.csv"),
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = FALSE
)

cat("文件已保存到", outdir)