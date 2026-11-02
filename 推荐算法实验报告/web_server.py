import importlib, json, os, sys, threading, traceback, webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
MY = os.path.join(ROOT, "data", "my_ratings.json")
PORT = int(os.environ.get("PORT", 8000))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

CTX = {}


def read_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def status():
    res = {}
    R = np.array([[5, 3, 0], [4, 0, 1], [0, 2, 5], [3, 3, 3]], float)
    for key, (name, mod) in CTX["algos"].items():
        mod = importlib.reload(mod)
        for fn in ("compute_similarity", "predict_rating"):
            try:
                S = mod.compute_similarity(R)
                if fn == "predict_rating":
                    mod.predict_rating(R, S, 0, 2, 2)
                res[f"{mod.__name__}.{fn}"] = "ok"
            except NotImplementedError:
                res[f"{mod.__name__}.{fn}"] = "todo"
            except Exception as e:
                res[f"{mod.__name__}.{fn}"] = f"error: {type(e).__name__}: {e}"
    return {"functions": res, "best_k": CTX["best_k"]}


def run_algo(key, mod, R, u, k):
    S = mod.compute_similarity(R)
    means = R.sum(axis=0) / np.maximum((R > 0).sum(axis=0), 1)     # 每部电影的评分均值（基于物品的公式 7.7′ 用）
    ranked = CTX["recommend_top_n"](mod, R, S, u, k, top_n=None)   # 调用 main.py 的 recommend_top_n()
    pred = dict(ranked)
    out = {}
    for i, p in pred.items():
        if key == "user":                         # 公式 7.5：评过 i 的用户里与你最像的 k 个
            cand = np.array([v for v in np.where(R[:, i] > 0)[0] if v != u], dtype=int)
            nb = cand[np.argsort(-S[u, cand])[:k]]
            rows = [{"user": int(v) + 1, "sim": float(S[u, v]), "r": float(R[v, i])} for v in nb]
        else:                                     # 公式 7.7′：你评过的电影里与 i 最像的 k 部，减去电影均值
            cand = np.array([j for j in np.where(R[u] > 0)[0] if j != i], dtype=int)
            nb = cand[np.argsort(-S[i, cand])[:k]]
            rows = [{"movie_id": int(j) + 1, "sim": float(S[i, j]), "r": float(R[u, j]), "mean": float(means[j]),
                     "dev": float(R[u, j] - means[j])} for j in nb]
        den = sum(x["sim"] for x in rows)
        if key == "user":
            num = sum(x["sim"] * x["r"] for x in rows)
            formula = num / den if den else 3.0
            base = None
        else:
            num = sum(x["sim"] * x["dev"] for x in rows)
            base = float(means[i])
            formula = float(np.clip(base + num / den, 1, 5)) if den else base
        d = {"score": p, "rows": rows, "num": num, "den": den, "pool": int(len(cand)),
             "formula": formula, "base": base}
        if key == "item":                         # 和这部电影最像的电影（全部 80 部里）
            top = np.argsort(-S[i])[:8]
            d["similar"] = [{"movie_id": int(j) + 1, "sim": float(S[i, j]), "mine": bool(R[u, j] > 0)} for j in top]
        out[str(i + 1)] = d
    recs = ranked[:10]
    res = {"k": k, "recs": [{"movie_id": i + 1, "score": p} for i, p in recs], "all": out,
           "curve": CTX["curves"].get(key)}
    if key == "user":                             # 和你口味最像的观众
        nbs = []
        common = ((R > 0) & (R[u] > 0)).sum(axis=1)
        for v in np.lexsort((-common, -S[u]))[:6]:      # 相似度相同时，共同评分多的排前面
            both = np.where((R[u] > 0) & (R[v] > 0))[0]
            nbs.append({"user": int(v) + 1, "sim": float(S[u, v]), "n_rated": int((R[v] > 0).sum()),
                        "common": [{"movie_id": int(j) + 1, "me": float(R[u, j]), "them": float(R[v, j])} for j in both]})
        res["neighbors"] = nbs
    return res


def recommend(mine):
    mine = {int(m): float(r) for m, r in mine.items() if 1 <= int(m) <= len(CTX["movies"]) and 1 <= float(r) <= 5}
    if len(mine) < 3:
        raise ValueError("至少给 3 部电影打分")
    with open(MY, "w", encoding="utf-8") as f:            # 评分保存成文件，算法从文件读取
        json.dump({str(m): r for m, r in mine.items()}, f, ensure_ascii=False, indent=2)
    mine = {int(m): r for m, r in read_json(MY, {}).items()}

    ratings = CTX["ratings"]
    me_id = int(ratings.user_id.max()) + 1                # "你" = 第 944 号用户
    me = pd.DataFrame({"user_id": me_id, "movie_id": list(mine), "rating": list(mine.values())})
    R = CTX["build_matrix"](pd.concat([ratings, me], ignore_index=True), me_id, len(CTX["movies"]))
    out = {"me": me_id, "n_users": me_id, "results": {}}
    for key, (name, mod) in CTX["algos"].items():
        if key not in CTX["best_k"]:
            out["results"][key] = {"error": f"{name}还没完成"}
            continue
        try:
            out["results"][key] = run_algo(key, importlib.reload(mod), R, me_id - 1, CTX["best_k"][key])
        except Exception as e:
            traceback.print_exc()
            out["results"][key] = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]}
    return out


class Handler(SimpleHTTPRequestHandler):
    # Windows 的注册表可能把文件类型登记错，这里固定下来
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                      ".css": "text/css; charset=utf-8", ".json": "application/json",
                      ".jpg": "image/jpeg", ".png": "image/png", ".woff2": "font/woff2"}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def log_message(self, fmt, *args):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(b)

    def fail(self, e):
        traceback.print_exc()
        self.send_json({"error": str(e) if isinstance(e, ValueError) else f"{type(e).__name__}: {e}",
                        "trace": traceback.format_exc()[-1500:]}, 500)

    def do_GET(self):
        try:
            if self.path.startswith("/api/movies"):
                cnt = CTX["ratings"].groupby("movie_id").rating.agg(["size", "mean"])
                colors = read_json(os.path.join(WEB, "posters", "colors.json"), {})
                out = []
                for m in CTX["movies"].itertuples():
                    p = f"posters/{m.movie_id:02d}.jpg"
                    out.append({"id": int(m.movie_id), "name": m.name, "en": m.en, "year": int(m.year),
                                "tags": m.genres.split("|") if isinstance(m.genres, str) else [],
                                "poster": p if os.path.exists(os.path.join(WEB, p)) else "",
                                "color": colors.get(str(m.movie_id), ""),
                                "n": int(cnt.loc[m.movie_id, "size"]), "avg": round(float(cnt.loc[m.movie_id, "mean"]), 2)})
                return self.send_json({"movies": out, "mine": read_json(MY, {})})
            if self.path.startswith("/api/status"):
                return self.send_json(status())
            return super().do_GET()
        except Exception as e:
            self.fail(e)

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if self.path.startswith("/api/recommend"):
                return self.send_json(recommend(body.get("ratings", {})))
            self.send_json({"error": "unknown api"}, 404)
        except Exception as e:
            self.fail(e)


class Server(ThreadingHTTPServer):
    allow_reuse_address = (os.name != "nt")   # Windows 上禁止两个程序共用同一端口
    daemon_threads = True


def run(algos, best_k, curves, build_matrix, recommend_top_n, movies, ratings):
    CTX.update(algos=algos, best_k=best_k, curves=curves, build_matrix=build_matrix,
               recommend_top_n=recommend_top_n, movies=movies, ratings=ratings)
    srv = None
    for port in range(PORT, PORT + 20):          # 8000 被占用就自动换下一个
        try:
            srv = Server(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if srv is None:
        print(f"端口 {PORT}~{PORT + 19} 都无法使用，网页启动失败")
        return
    url = f"http://127.0.0.1:{port}"
    print(f"       网页地址：{url}  （浏览器没有自动打开的话，把这个地址复制到浏览器）")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
