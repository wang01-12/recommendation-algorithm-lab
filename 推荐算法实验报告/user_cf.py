import numpy as np


def compute_similarity(rating_matrix):
    R = np.asarray(rating_matrix, dtype=float)
    rated = R > 0                                   # 每个用户评过分的电影（布尔掩码）
    n_users = R.shape[0]
    similarity_matrix = np.zeros((n_users, n_users))
    for a in range(n_users):
        mask_a = rated[a]
        if not mask_a.any():                        # 该用户一条评分都没有，与任何人相似度都是 0
            continue
        common = mask_a & rated                     # (用户数, 电影数)：两人都评过分的电影
        has_common = common.any(axis=1)             # 是否存在共同评分的电影
        diff = R[a] - R                             # 每个用户与 a 在各电影上的评分差
        d = np.sqrt((diff ** 2 * common).sum(axis=1))   # 只在共同评分的电影上求欧式距离 (7.1)
        s = 1.0 / (1.0 + d)                         # 距离转相似度 (7.2)
        s[~has_common] = 0.0                        # 没有共同评分的电影 → 相似度 0
        similarity_matrix[a] = s
    np.fill_diagonal(similarity_matrix, 0.0)        # 用户与自己的相似度为 0
    return similarity_matrix


def predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors):
    col = rating_matrix[:, movie_index]             # 所有用户对这部电影的评分
    candidates = np.where(col > 0)[0]               # 评过这部电影的用户
    candidates = candidates[candidates != user_index]   # 去掉自己
    if len(candidates) == 0:
        return 3.0
    sims = similarity_matrix[candidates, user_index]    # 这些用户与该用户的相似度
    top = candidates[np.argsort(-sims, kind="stable")[:k_neighbors]]   # 相似度最高的 K 个近邻（并列时按下标先后）
    top_sims = similarity_matrix[top, user_index]
    if top_sims.sum() == 0:                         # 相似度之和为 0 → 预测分 3
        return 3.0
    predicted_rating = (top_sims * rating_matrix[top, movie_index]).sum() / top_sims.sum()   # 公式 (7.5)
    return float(predicted_rating)
