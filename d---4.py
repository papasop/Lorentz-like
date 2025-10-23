# ==== DAG–Ward: d→4 收敛验证（对数分箱·中位数·稳健拟合 / PBC / 恒定密度）====
# 复制到 Colab 单元格直接运行

import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from numpy.random import default_rng
rng = default_rng(20251023)

# ---------------- 参数（可按需调） ----------------
alpha       = 0.9          # 纵横比：Lsp = alpha * Tmax
alpha_scale = 1.8          # 体积放宽比例（1.6~2.0，越大边界偏差越小）
Ns          = [5000, 8000, 10000, 15000, 20000, 30000]

# 固有时窗口（“中段”）
tau_lo_f, tau_hi_f = 0.20, 0.70

# 候选 (p,q) 对数（越大越稳）
pairs_per_conf = 200_000

# 对数分箱个数（20~30）
n_bins = 24
min_bin_count = 200   # 每个有效分箱最少样本要求

# 回归设置（对“分箱后的中位数”做回归）
trim_frac = 0.08
huber_c   = 1.345
huber_iter= 60

# 输出
out_dir = "/mnt/data"; os.makedirs(out_dir, exist_ok=True)

# ---------------- 工具函数 ----------------
def minimal_image(dx, L):
    # 空间 PBC: [-L/2, L/2]
    return dx - L * np.round(dx / L)

def sprinkling(N, Tmax, Lsp):
    t = rng.uniform(0.0, Tmax, N)
    x = rng.uniform(-Lsp/2, Lsp/2, (N,3))
    return t, x

def huber_fit(x, y, c=1.345, iters=60, tol=1e-8):
    # y = a x + b 的 Huber IRLS 稳健回归（对分箱后的点，样本量较小）
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(iters):
        r = y - X @ beta
        s = np.median(np.abs(r)) / 0.6745 + 1e-12
        u = r / (c * s)
        w = np.where(np.abs(u) > 1, 1/np.abs(u), 1.0)
        WX = X * w[:,None]; Wy = y * w
        beta_new, *_ = np.linalg.lstsq(WX, Wy, rcond=None)
        if np.linalg.norm(beta_new - beta) < tol * (1 + np.linalg.norm(beta)):
            beta = beta_new; break
        beta = beta_new
    return float(beta[0])

# 设定基准密度（恒定）
N0, T0 = 3000, 1.0
rho = N0 / (T0 * (alpha*T0)**3)

def estimate_D_for_config(N):
    # 1) 构造几何（恒定密度 + 体积放宽 + 固定纵横比）
    Tmax = alpha_scale * (N / (rho * (alpha**3)))**0.25
    Lsp  = alpha_scale * alpha * Tmax
    t, x = sprinkling(N, Tmax, Lsp)

    # 2) 候选对（向量化生成）
    p = rng.integers(0, N-1, size=pairs_per_conf)
    q = rng.integers(1, N,   size=pairs_per_conf)
    mask_fwd = t[q] > t[p]
    p, q = p[mask_fwd], q[mask_fwd]
    if p.size == 0:
        return np.nan, 0, Tmax, Lsp

    dt = t[q] - t[p]
    dx = minimal_image(x[q] - x[p], Lsp)
    s2 = dt*dt - np.sum(dx*dx, axis=1)
    mask_timelike = s2 > 0
    p, q, s2 = p[mask_timelike], q[mask_timelike], s2[mask_timelike]
    if p.size == 0:
        return np.nan, 0, Tmax, Lsp

    tau = np.sqrt(s2)
    tau_lo, tau_hi = tau_lo_f * Tmax, tau_hi_f * Tmax
    mask_tau = (tau > tau_lo) & (tau < tau_hi)
    p, q, tau = p[mask_tau], q[mask_tau], tau[mask_tau]
    if p.size == 0:
        return np.nan, 0, Tmax, Lsp

    # 3) 计算区间体积 |I(p,q)|（度量因果：p->k 与 k->q 均类时）
    #    为避免 O(N^2) 内存，这里逐对做，但只留下 (tau, |I|) 即可
    taus, sizes = [], []
    for pi, qi, taui in zip(p, q, tau):
        mask_k = (t > t[pi]) & (t < t[qi])
        idx = np.where(mask_k)[0]
        if idx.size < 3: 
            continue
        dt1 = t[idx] - t[pi]; dx1 = minimal_image(x[idx] - x[pi], Lsp)
        c1  = dt1*dt1 - np.sum(dx1*dx1, axis=1) > 0
        dt2 = t[qi] - t[idx]; dx2 = minimal_image(x[qi] - x[idx], Lsp)
        c2  = dt2*dt2 - np.sum(dx2*dx2, axis=1) > 0
        mI  = int(np.sum(c1 & c2))
        if mI >= 2:
            taus.append(taui); sizes.append(mI)

    taus = np.array(taus); sizes = np.array(sizes)
    if taus.size < n_bins * min_bin_count:
        # 样本不足，不做拟合
        return np.nan, int(taus.size), Tmax, Lsp

    # 4) 对数分箱：bin 中位数（减弱异方差与尾部稀疏的偏差）
    #    注意：对 tau 做对数等距分箱
    log_tau = np.log(taus + 1e-300)
    log_lo, log_hi = np.log(tau_lo + 1e-300), np.log(tau_hi + 1e-300)
    edges = np.linspace(log_lo, log_hi, n_bins + 1)

    bin_centers, bin_medians, bin_counts = [], [], []
    for i in range(n_bins):
        mask_bin = (log_tau >= edges[i]) & (log_tau < edges[i+1])
        if np.sum(mask_bin) >= min_bin_count:
            # 该 bin 的 tau 中心取几何均值、|I| 取中位数
            tau_center = np.exp(0.5 * (edges[i] + edges[i+1]))
            bin_centers.append(tau_center)
            bin_medians.append(np.median(sizes[mask_bin]))
            bin_counts.append(int(np.sum(mask_bin)))

    bin_centers = np.array(bin_centers); bin_medians = np.array(bin_medians)
    if bin_centers.size < max(8, n_bins//3):
        return np.nan, int(taus.size), Tmax, Lsp

    # 5) 对“分箱后的点”做 log–log 拟合（Huber + 中央修剪）
    xv = np.log(bin_centers + 1e-300); yv = np.log(bin_medians + 1e-300)
    order = np.argsort(xv); xv, yv = xv[order], yv[order]
    n = len(xv)
    lo = int(np.floor(trim_frac * n)); hi = int(np.ceil((1 - trim_frac) * n))
    xv, yv = xv[lo:hi], yv[lo:hi]
    if xv.size < 6:
        return np.nan, int(taus.size), Tmax, Lsp

    D_hat = huber_fit(xv, yv, c=huber_c, iters=huber_iter)
    return float(D_hat), int(taus.size), Tmax, Lsp

# ---------------- 主流程 ----------------
rows = []
for N in Ns:
    D_hat, n_pairs, Tmax, Lsp = estimate_D_for_config(N)
    rows.append({"N":N, "Tmax":Tmax, "Lsp":Lsp, "D_est":D_hat, "pairs_used":n_pairs})
    print(f"N={N:5d} | Tmax={Tmax:.3f} Lsp={Lsp:.3f} | D≈{D_hat:.3f} | pairs={n_pairs}")

df = pd.DataFrame(rows)
csv_path  = os.path.join(out_dir, "d4_logbin_median_huber.csv")
plot_path = os.path.join(out_dir, "d4_logbin_median_huber.png")
df.to_csv(csv_path, index=False)

plt.figure()
plt.plot(df["N"], df["D_est"], "o-")
plt.axhline(4.0, ls="--")
plt.xlabel("N"); plt.ylabel("D_est (Huber on log-binned medians)")
plt.title(f"DAG–Ward: d→4 with PBC, log-binning medians  (alpha_scale={alpha_scale})")
plt.savefig(plot_path, bbox_inches='tight'); plt.close()

print("Saved CSV:", csv_path)
print("Saved PNG:", plot_path)
