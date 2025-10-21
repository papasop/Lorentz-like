# === Time-series K (no subsample artifact): Free & Weakly-driven ===
# B = D^T W_E D  (same W_E for Flux & EOM), Flux = -D^T W_E D φ
# Outputs: Ward (2 cases), K_time (2 cases), leak, c_eff + simple dim proxy
# Only numpy/matplotlib; plt+print only.

import numpy as np, matplotlib.pyplot as plt, math

# ---------- Causal DAG in 4D diamond ----------
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

def incidence_matrix(N,E):
    M=len(E); D=np.zeros((M,N))
    for k,(i,j) in enumerate(E): D[k,i]=-1.0; D[k,j]=+1.0
    return D

def build_B_exact(D, w_e):
    # B = D^T W_E D
    return D.T @ (w_e[:,None]*D)

def spectral_radius_est(B, iters=24, seed=0):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=B.shape[0]); x/=np.linalg.norm(x)+1e-12
    for _ in range(iters):
        y=B@x; n=np.linalg.norm(y)
        if n<1e-14: break
        x=y/(n+1e-12)
    return float(np.linalg.norm(B@x))

# ---------- Time evolution returning full series ----------
def evolve_series(N=1600, CFL=0.025, steps=360, seed=1, lam=0.0, we_scheme="abs_dt"):
    # graph
    t,X = sample_in_diamond(N, 1.0, 1.0, seed=seed)
    E, dt_e, dx_e = build_edges_eps(t, X, eps=1e-9)
    D = incidence_matrix(N, E)

    # edge weights
    if we_scheme=="abs_dt":
        w_e = np.abs(dt_e) + 1e-12
    else:
        w_e = np.ones_like(dt_e)

    B = build_B_exact(D, w_e)

    rho = spectral_radius_est(B, iters=24, seed=seed)
    dt = CFL/(rho + 1e-12)

    rng=np.random.default_rng(seed+7)
    phi = rng.normal(size=N)*0.8
    v   = np.zeros_like(phi)

    ward_rel=[]; Phi_series=[]; H_series=[]

    for _ in range(steps):
        v   = v + dt*(B@phi) - dt*lam*np.clip(phi,-6,6)**3
        phi = phi + dt*v

        Dphi = D@phi
        Flux = - D.T @ (w_e * Dphi)
        EOM  = B @ phi - lam*np.clip(phi,-6,6)**3

        fn=np.linalg.norm(Flux); en=np.linalg.norm(EOM); rn=np.linalg.norm(Flux+EOM)
        denom = fn + en + 1e-12
        ward_rel.append(rn/denom)

        Phi_series.append(np.sum(np.abs(phi)))
        H_series.append(np.sum(phi*phi))

    return {
        "ward": np.array(ward_rel),
        "Phi":  np.array(Phi_series),
        "H":    np.array(H_series),
        "phi":  phi,
        "t": t, "X": X, "E": E, "dt_e": dt_e, "dx_e": dx_e
    }

# ---------- Robust time-based K estimators ----------
def huber_fit(x,y,delta=1.0,iters=25):
    A=np.vstack([x,np.ones_like(x)]).T; w=np.ones_like(x)
    for _ in range(iters):
        Aw=A*w[:,None]; yw=y*w
        a,b=np.linalg.lstsq(Aw,yw,rcond=None)[0]
        r=y-(a*x+b); s=max(1e-12, np.median(np.abs(r))*1.4826)
        w=1/np.maximum(1, np.abs(r/(delta*s)))
    return float(a), float(b)

def K_time_huber(Phi, H, drop_frac=0.30, win=56):
    T=len(Phi); s=int(T*drop_frac)
    Ks=[]
    for t in range(s, T):
        i0=max(s, t-win+1)
        x=np.log(Phi[i0:t+1]+1e-12); y=np.log(H[i0:t+1]+1e-12)
        if len(x)<12: Ks.append(np.nan); continue
        a,_=huber_fit(x,y,delta=1.0); Ks.append(a)
    return np.array(Ks)

def K_time_binned(Phi, H, drop_frac=0.30, nbins=14, boot=200):
    T=len(Phi); s=int(T*drop_frac)
    x=np.log(Phi[s:]+1e-12); y=np.log(H[s:]+1e-12)
    qs=np.linspace(0,1,nbins+1); edges=np.quantile(x,qs)
    xb=[]; yb=[]
    for i in range(nbins):
        sel=(x>=edges[i])&(x<edges[i+1])
        if np.sum(sel)<max(6,int(0.03*len(x))): continue
        xb.append(np.median(x[sel])); yb.append(np.median(y[sel]))
    xb=np.array(xb); yb=np.array(yb)
    if len(xb)<5: return np.nan, (np.nan,np.nan)
    a,_=np.polyfit(xb,yb,1)
    slopes=[]
    for _ in range(boot):
        idx=np.random.randint(0,len(xb),size=len(xb))
        a2,_=np.polyfit(xb[idx], yb[idx], 1); slopes.append(a2)
    lo,hi=np.percentile(slopes,[16,84])
    return float(a), (float(lo), float(hi))

# ---------- Lightcone leak & c_eff ----------
def leak_metrics(t,X):
    n_strict   = len(build_edges_eps(t,X,eps=1e-9)[0])
    n_tighter  = len(build_edges_eps(t,X,eps=1e-12)[0])
    n_relaxed  = len(build_edges_eps(t,X,eps=1e-3)[0])  # 更严的“relaxed”取值
    return 1.0 - n_tighter/max(1,n_strict), (n_relaxed - n_strict)/max(1,n_relaxed)

def c_eff_near_null(dt_e, dx_e, q=0.06):
    s = np.abs(dt_e*dt_e - dx_e*dx_e)
    if len(s)==0: return np.nan
    k = max(12, int(len(s)*q))
    idx = np.argsort(s)[:k]
    return float(np.median(dt_e[idx]/(dx_e[idx]+1e-12)))

# ---------- Simple dimension proxy (coarse) ----------
def dim_proxy(t,X,E, M=600, seed=3):
    rng=np.random.default_rng(seed); N=len(t); vals=[]
    for _ in range(M):
        a,b=rng.integers(0,N,2)
        if t[b]<=t[a]: continue
        dt_=t[b]-t[a]; r_=np.linalg.norm(X[b]-X[a]); s2=dt_*dt_-r_*r_
        if s2<=0: continue
        vals.append(np.log(s2+1e-12))
    if len(vals)<8: return np.nan
    y=np.sort(np.array(vals)); x=np.arange(len(y))
    a,_=np.polyfit(x,y,1); d=1.0/max(1e-12,abs(a))
    return float(d)

# ==================== RUN: Free & Weakly-driven ====================
cfg = dict(N=1600, CFL=0.025, steps=360, we_scheme="abs_dt")
out_free   = evolve_series(seed=21, lam=0.0, **cfg)
out_drive  = evolve_series(seed=37, lam=5e-3, **cfg)

# Ward plots
plt.figure(figsize=(6.6,3.6))
plt.plot(out_free["ward"],  label="free (λ=0)")
plt.plot(out_drive["ward"], label="driven (λ=5e-3)")
plt.xlabel("step"); plt.ylabel("WardRel"); plt.title("Ward residual over time (t-series)")
plt.legend(); plt.show()

# Time-based K
Kf = K_time_huber(out_free["Phi"], out_free["H"], drop_frac=0.30, win=56)
Kd = K_time_huber(out_drive["Phi"],out_drive["H"],drop_frac=0.30, win=56)

plt.figure(figsize=(6.6,3.6))
plt.plot(Kf, label="free")
plt.plot(Kd, label="driven")
plt.axhline(2.0, ls="--", lw=1, alpha=0.7)
plt.xlabel("tail index"); plt.ylabel("K (Huber, sliding)")
plt.title("Scaling K from time-series (no subsample artifact)")
plt.legend(); plt.show()

# Coarse dim proxy (just to watch trend)
d_est = dim_proxy(out_free["t"], out_free["X"], out_free["E"], M=600, seed=5)

# Leak & c_eff
leak_s, leak_r = leak_metrics(out_free["t"], out_free["X"])
c_eff = c_eff_near_null(out_free["dt_e"], out_free["dx_e"])

# Summary
def qstats(a): return np.median(a), np.percentile(a,10), np.percentile(a,90)
wf, wf10, wf90 = qstats(out_free["ward"]);  wd, wd10, wd90 = qstats(out_drive["ward"])
Kf_med = np.nanmedian(Kf);  Kd_med = np.nanmedian(Kd)
Kf_bin,(Kf_lo,Kf_hi) = K_time_binned(out_free["Phi"], out_free["H"])
Kd_bin,(Kd_lo,Kd_hi) = K_time_binned(out_drive["Phi"], out_drive["H"])

print("== SUMMARY (time-series, artifact-free K) ==")
print(f"Ward (free):   median={wf:.3e}, p10={wf10:.3e}, p90={wf90:.3e}")
print(f"Ward (driven): median={wd:.3e}, p10={wd10:.3e}, p90={wd90:.3e}")
print(f"K (free, Huber sliding)  median={Kf_med:.3f}")
print(f"K (driven, Huber sliding) median={Kd_med:.3f}")
print(f"K (free, binned):  {Kf_bin:.3f}  (68% CI [{Kf_lo:.3f}, {Kf_hi:.3f}])")
print(f"K (driven, binned): {Kd_bin:.3f}  (68% CI [{Kd_lo:.3f}, {Kd_hi:.3f}])")
print(f"Dim proxy ≈ {d_est:.3f}")
print(f"Leak strict={leak_s:.3e}, relaxed={leak_r:.3e};  c_eff≈{c_eff:.3f}")
