import os, time
import numpy as np
import pandas as pd
import user_cf
import item_cf

ROOT = os.path.dirname(os.path.abspath(__file__))
ALGORITHMS = {"user": ("基于用户的协同过滤", user_cf),
              "item": ("基于物品的协同过滤", item_cf)}
CANDIDATE_K = {"user": [5, 10, 20, 30, 50, 80, 120, 200, 400],    # 候选的近邻个数 K
          "item": [2, 3, 5, 8, 10, 15, 20, 30, 50]}


def build_matrix(ratings, num_users, num_movies):
    rating_matrix = np.zeros((num_users, num_movies))
    for user_id, movie_id, rating in zip(ratings.user_id, ratings.movie_id, ratings.rating):
        rating_matrix[user_id - 1, movie_id - 1] = rating
    return rating_matrix


def choose_best_k(algorithm, rating_matrix, test_set, candidate_k_list):

    similarity_matrix = algorithm.compute_similarity(rating_matrix)

    user_index_list = test_set.user_id.values - 1
    movie_index_list = test_set.movie_id.values - 1
    true_ratings = test_set.rating.values.astype(float)

    rmse_list = []
    for k_neighbors in candidate_k_list:
        # 3. 对每个候选 K，逐条预测测试集的"考题"
        predictions = np.array([
            algorithm.predict_rating(rating_matrix, similarity_matrix, u, m, k_neighbors)
            for u, m in zip(user_index_list, movie_index_list)
        ])
        # 4. 均方根误差 RMSE
        rmse_list.append(float(np.sqrt(((true_ratings - predictions) ** 2).mean())))

    # 5. RMSE 最小的 K 就是最优 K
    best_k = int(candidate_k_list[int(np.argmin(rmse_list))])
    return best_k, rmse_list


def recommend_top_n(algorithm, rating_matrix, similarity_matrix, user_index, k_neighbors, top_n=10):

    unseen_movies = np.where(rating_matrix[user_index] == 0)[0]          # 该用户没评过分的电影
    predictions = [(int(movie_index),
                    float(algorithm.predict_rating(rating_matrix, similarity_matrix, user_index, movie_index, k_neighbors)))
                   for movie_index in unseen_movies]                       # 逐部预测评分
    predictions.sort(key=lambda item: item[1], reverse=True)               # 预测评分从高到低排序
    return predictions if top_n is None else predictions[:top_n]


def main():

    movies = pd.read_csv(os.path.join(ROOT, "data", "movies.csv"), encoding="utf-8")
    ratings = pd.read_csv(os.path.join(ROOT, "data", "ratings.csv"))
    num_users, num_movies = ratings.user_id.max(), len(movies)
    print(f"第1步  读取数据：{num_users} 位用户，{num_movies} 部电影，{len(ratings)} 条评分")

    shuffled = ratings.sample(frac=1, random_state=42)
    cut = int(len(shuffled) * 0.8)
    train_set, test_set = shuffled[:cut], shuffled[cut:]
    print(f"第2步  训练集 {len(train_set)} 条，测试集 {len(test_set)} 条（测试集的分数先藏起来，用来检验预测准不准）")

    rating_matrix = build_matrix(train_set, num_users, num_movies)
    print(f"第3步  评分矩阵：{rating_matrix.shape[0]} 行（用户）× {rating_matrix.shape[1]} 列（电影），"
          f"有评分的格子只占 {(rating_matrix > 0).mean():.0%}")

    full_matrix = build_matrix(ratings, num_users, num_movies)   # 全部评分的矩阵，推荐测试时使用
    movie_names = dict(zip(movies.movie_id - 1, movies.name))

    best_k_of, curves = {}, {}
    for key, (name, algorithm) in ALGORITHMS.items():
        print(f"\n========== {name} ==========")

        try:
            t = time.time()
            best_k, rmse_list = choose_best_k(algorithm, rating_matrix, test_set, CANDIDATE_K[key])
        except NotImplementedError as e:
            print(f"  还没完成：{e}（先跳过）")
            continue
        print(f"第4步  选择近邻个数 K（用时 {time.time() - t:.1f} 秒）：")
        for k_neighbors, rmse in zip(CANDIDATE_K[key], rmse_list):
            print(f"         K = {k_neighbors:<4d} 测试集 RMSE = {float(rmse):.4f}")
        print(f"       → 最优 K = {best_k}，RMSE = {min(rmse_list):.4f}")
        best_k_of[key] = int(best_k)
        curves[key] = {"k": CANDIDATE_K[key], "rmse": [round(float(e), 4) for e in rmse_list], "best": int(best_k)}

        similarity_matrix = algorithm.compute_similarity(full_matrix)
        top5 = recommend_top_n(algorithm, full_matrix, similarity_matrix, user_index=0, k_neighbors=best_k, top_n=5)
        print("       测试 recommend_top_n()：给 1 号用户推荐的 5 部电影")
        for rank, (movie_index, score) in enumerate(top5, 1):
            print(f"         {rank}. 《{movie_names[movie_index]}》  预测评分 {score:.2f}")


    if not best_k_of:
        print("\n还没有算法完成第4步，网页暂时打不开。")
        return
    print("\n第5步  打开推荐网页（按 Ctrl+C 结束）")
    import web_server
    web_server.run(ALGORITHMS, best_k_of, curves, build_matrix, recommend_top_n, movies, ratings)


if __name__ == "__main__":
    main()
