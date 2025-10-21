# === Alexandrov d → 4: "only increase M" vs "only increase depth" (single DAG) ===
# - Build one 4D-diamond DAG (N=4000) once
# - Estimate d with bounded BFS + binned regression + bootstrap
# - Three configs: baseline (depth=10, M=1000), M-only↑ (M=1500), depth-only↑ (depth=12)
# - Only numpy/matplotlib; print + plt.show(); no file I/O.

import numpy as np, matplotlib.pyplot as plt
from collections import deque
import time

# ---------------- Causal DAG in 4D diamond ----------------
def sample_in_diamond(N, Rt=1.0, Rx=1.0, seed=0):
    rng=np.random.default_rng(seed)
    tL=[]; XL=[]
    while len(tL)<N:
        t=rng.uniform(-Rt,Rt)
        x=rng.uniform(-Rx,Rx,size=3)
        if np.linalg.norm(x)>Rx: continue
        if abs(t)/Rt + np.linalg.norm(x)/Rx <= 1.0:
            tL.append(t); XL.append(x)
    return np.array(tL), np.vstack(XL)

def build_edges_eps(t, X, eps=1e-9):
    N=len(t); order=np.argsort(t); ts=t[order]; Xs=X[order]
    E=[]; dt=[]; dx=[]
    for a in range(N):
        ta=ts[a]; Xa=Xs[a]
        for b in range(a+1,N):
            dtt=ts[b]-ta
            if dtt<=0: continue
            r=np.linalg.norm(Xs[b]-Xa)
            if dtt*dtt - r*r >= -eps:
                E.append((order[a],order[b])); dt.append(dtt); dx.append(r)
    return np.array(E,int), np.array(dt), np.array(dx)

def build_adj(N,E):
    fwd=[[] for _ in range(N)]; bwd=[[] for _ in range(N)]
    for i,j in E:
        fwd[i].append(j); bwd[j].append(i)
    return fwd,bwd

# ---------------- Bounded BFS (depth-limited) ----------------
def bfs_limited(start, adj, depth, mask=None):
    seen=set([start]); q=deque([(start,0)])
    out=set()
    while q:
        u,d=q.popleft()
        if d==depth: continue
        for v in adj[u]:
            if mask is not None and (not mask[v]): continue
            if v in seen: continue
            seen.add(v)
            out.add(v)
            q.append((v,d+1))
    return out

# ---------------- Alexandrov d (binned + bootstrap) ----------------
def alexandrov_dim(t, X, E, depth=10, M=1000, interior_frac=0.75,
                   nbins=16, boot=400, seed=0):
    rng=np.random.default_rng(seed)
    N=len(t)
    # interior mask （去边界）
    tmin,tmax=np.min(t),np.max(t); span=tmax-tmin
    t_lo=tmin + (1-interior_frac)/2*span
    t_hi=tmax - (1-interior_frac)/2*span
    mask=(t>=t_lo)&(t<=t_hi)

    fwd,bwd = build_adj(N,E)
    logs_tau=[]; logs_I=[]

    tries=0
    while len(logs_tau)<M and tries<5*M:
        p,q = rng.integers(0,N,2)
        tries+=1
        if t[q] <= t[p]: continue
        dt = t[q]-t[p]; r = np.linalg.norm(X[q]-X[p]); s2 = dt*dt - r*r
        if s2 <= 0: continue
        if (not mask[p]) or (not mask[q]): continue

        Fp = bfs_limited(p, fwd, depth, mask)
        Pq = bfs_limited(q, bwd, depth, mask)
        I  = len(Fp.intersection(Pq))
        if I<=0: continue

        logs_tau.append(np.log(np.sqrt(s2)+1e-12))
        logs_I.append(np.log(I+1e-12))

    logs_tau=np.array(logs_tau); logs_I=np.array(logs_I)
    if len(logs_tau) < max(50, nbins):
        return np.nan, (np.nan, np.nan), (logs_tau, logs_I, None, None)

    # 对数分箱中位数拟合
    qs=np.linspace(0,1,nbins+1); edges=np.quantile(logs_tau, qs)
    xb=[]; yb=[]
    for i in range(nbins):
        sel=(logs_tau>=edges[i])&(logs_tau<edges[i+1])
        if np.sum(sel)<max(10,int(0.01*len(logs_tau))): continue
        xb.append(np.median(logs_tau[sel])); yb.append(np.median(logs_I[sel]))
    xb=np.array(xb); yb=np.array(yb)
    if len(xb)<5:
        return np.nan, (np.nan, np.nan), (logs_tau, logs_I, None, None)

    a,_=np.polyfit(xb,yb,1); d=float(a)  # slope ~ effective dimension
    # bootstrap CI
    slopes=[]
    for _ in range(boot):
        idx=np.random.randint(0,len(xb),size=len(xb))
        a2,_=np.polyfit(xb[idx], yb[idx], 1)
        slopes.append(a2)
    lo,hi=np.percentile(slopes,[16,84])
    return d, (float(lo), float(hi)), (logs_tau, logs_I, xb, yb)

# ---------------- Lightcone & c_eff sanity (可选) ----------------
def leak_metrics(t,X):
    n_strict   = len(build_edges_eps(t,X,eps=1e-9)[0])
    n_tighter  = len(build_edges_eps(t,X,eps=1e-12)[0])
    n_relaxed  = len(build_edges_eps(t,X,eps=1e-3)[0])
    return 1.0 - n_tighter/max(1,n_strict), (n_relaxed - n_strict)/max(1,n_relaxed)

def c_eff_near_null(dt_e, dx_e, q=0.06):
    s = np.abs(dt_e*dt_e - dx_e*dx_e)
    k = max(12, int(len(s)*q))
    idx = np.argsort(s)[:k]
    return float(np.median(dt_e[idx]/(dx_e[idx]+1e-12)))

# ===================== Build one DAG (N=4000) =====================
N=4000; seed=20251022
t0=time.time()
t,X = sample_in_diamond(N, 1.0, 1.0, seed=seed)
E, dt_e, dx_e = build_edges_eps(t, X, eps=1e-9)
t1=time.time()
print(f"[Build] N={N}, edges={len(E)}, time={t1-t0:.1f}s")

# Quick sanity
leak_s, leak_r = leak_metrics(t,X)
ceff = c_eff_near_null(dt_e, dx_e)
print(f"[Lightcone/Lorentz] Leak strict={leak_s:.3e}, relaxed={leak_r:.3e};  c_eff≈{ceff:.3f}")

# ===================== Three configurations =====================
cfgs = [
    ("baseline (depth=10, M=1000)", dict(depth=10, M=1000)),
    ("only-M↑ (depth=10, M=1500)", dict(depth=10, M=1500)),
    ("only-depth↑ (depth=12, M=1000)", dict(depth=12, M=1000)),
]

results = []
for name, kw in cfgs:
    print(f"\n[Run] {name}")
    t2=time.time()
    d_hat, (d_lo,d_hi), (logs_tau, logs_I, xb, yb) = alexandrov_dim(
        t, X, E, depth=kw["depth"], M=kw["M"],
        interior_frac=0.75, nbins=16, boot=400, seed=seed
    )
    t3=time.time()
    results.append((name, kw["depth"], kw["M"], d_hat, d_lo, d_hi, t3-t2))

    # 可视化
    if xb is not None and yb is not None:
        plt.figure(figsize=(6.3,4.4))
        plt.scatter(logs_tau, logs_I, s=4, alpha=0.12, label="pairs")
        plt.plot(xb, yb, lw=2.2, label="bin medians")
        plt.xlabel("log τ"); plt.ylabel("log |I(p,q)|")
        plt.title(f"Alexandrov scaling — {name}")
        plt.legend(); plt.tight_layout(); plt.show()

# ===================== Summary =====================
print("\n== SUMMARY: Alexandrov dimension (single DAG) ==")
for name, depth, M, d_hat, d_lo, d_hi, dt in results:
    print(f"{name:28s}  d = {d_hat:.3f}  (68% CI [{d_lo:.3f}, {d_hi:.3f}])   time={dt:.1f}s")

print("\n[Tip] 若还需进一步推近 4：优先把 depth→11–12 或 M→1800（单独提升），"
      "并把 interior_frac→0.80 以更强去边界；注意运行时间会增加。")
