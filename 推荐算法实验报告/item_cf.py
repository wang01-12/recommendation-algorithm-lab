import numpy as np


def compute_similarity(rating_matrix):
    R = np.asarray(rating_matrix, dtype=float).T    # 转置后每行是一部电影，写法与用户版完全一致
    rated = R > 0                                   # 每部电影被评过分的用户（布尔掩码）
    n_movies = R.shape[0]
    similarity_matrix = np.zeros((n_movies, n_movies))
    for i in range(n_movies):
        mask_i = rated[i]
        if not mask_i.any():                        # 该电影一条评分都没有，相似度全为 0
            continue
        common = mask_i & rated                     # (电影数, 用户数)：两部电影都被评过分的用户
        has_common = common.any(axis=1)             # 是否存在共同评分的用户
        diff = R[i] - R                             # 每部电影与 i 在各用户上的评分差
        d = np.sqrt((diff ** 2 * common).sum(axis=1))   # 只在共同评分的用户上求欧式距离 (7.1)
        s = 1.0 / (1.0 + d)                         # 距离转相似度 (7.2)
        s[~has_common] = 0.0                        # 没有共同评分的用户 → 相似度 0
        similarity_matrix[i] = s
    np.fill_diagonal(similarity_matrix, 0.0)        # 电影与自己的相似度为 0
    return similarity_matrix


def predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors):
    R = np.asarray(rating_matrix, dtype=float)
    counts = (R > 0).sum(axis=0)                    # 每部电影被评过的次数
    means = np.where(counts > 0, R.sum(axis=0) / np.maximum(counts, 1), 0.0)   # 各电影的评分均值（0 不算）
    mean_i = means[movie_index] if counts[movie_index] > 0 else 3.0

    row = R[user_index]                             # 该用户对所有电影的评分
    candidates = np.where(row > 0)[0]               # 该用户评过的电影
    candidates = candidates[candidates != movie_index]   # 去掉目标电影本身
    if len(candidates) == 0:                        # 没有可用的近邻电影 → 预测分为均值
        return float(np.clip(mean_i, 1, 5))
    sims = similarity_matrix[candidates, movie_index]    # 这些电影与目标电影的相似度
    top = candidates[np.argsort(-sims, kind="stable")[:k_neighbors]]   # 相似度最高的 K 部近邻电影（并列时按下标先后）
    top_sims = similarity_matrix[top, movie_index]
    if top_sims.sum() == 0:                         # 相似度之和为 0 → 预测分为均值
        return float(np.clip(mean_i, 1, 5))
    deviation = row[top] - means[top]               # 用户对近邻电影的评分比其均值高多少
    predicted_rating = mean_i + (top_sims * deviation).sum() / top_sims.sum()   # 公式 (7.7′)
    return float(np.clip(predicted_rating, 1, 5))   # 超出 1~5 时截断
