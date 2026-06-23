# metrics_utils.py
# 纯聚类指标计算工具：仅返回ARI和NMI，无KMeans依赖，通用所有聚类算法
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compute_clustering_metrics(y_true, y_pred):
    """
    计算聚类的核心评估指标：ARI（调整兰德指数）、NMI（归一化互信息）
    :param y_true: 真实标签数组 (n_samples,)
    :param y_pred: 聚类预测标签数组 (n_samples,)
    :return: ari(保留4位小数), nmi(保留4位小数)
    """
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    # 保留4位小数，和你原有日志输出格式一致
    return round(ari, 4), round(nmi, 4)