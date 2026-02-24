
# ============================================================
# 11) Andreasen Sharpe-ratio diagnostics: N, LN, SR + plots
#    + extra sanity checks (mu norm, grad norm, drift term)
#    + extra plots (mu_norm over time + hist)
#    + L (Cholesky) plots + printed matrices
# ============================================================

from Code.utils.sharpe_ratio import SR_andreasen_reference


# ---------- helpers ----------
@torch.no_grad()
def pick_one_curve_per_currency_on_date(meta_df: pd.DataFrame, date_pick):
    m = meta_df.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"])
    date_pick = pd.to_datetime(date_pick)

    sel = m[m["as_of_date"] == date_pick].copy()
    if sel.empty:
        raise ValueError(f"No rows in meta_df for date {date_pick.date()}")

    sel = sel.sort_values(["ccy", "as_of_date"]).drop_duplicates(subset=["ccy"], keep="last")
    return sel.index.to_numpy(), sel


def plot_mu_norm_over_time_and_hist(params_df: pd.DataFrame, mu_cols, cfg: H.PlotConfig):
    """
    Adds mu_norm = sqrt(sum_k mu_k^2) then plots over time and histogram.
    Uses your existing helper functions: H.plot_param_over_time, H.hist_param
    """
    mu_cols = [c for c in mu_cols if c in params_df.columns]
    if len(mu_cols) == 0:
        print("No mu columns found for mu_norm plot. Skipping.")
        return

    mu_sq = None
    for c in mu_cols:
        v = params_df[c].astype(float).values
        mu_sq = v * v if mu_sq is None else (mu_sq + v * v)

    dfp = params_df.copy()
    dfp["mu_norm"] = np.sqrt(mu_sq)

    H.plot_param_over_time(dfp, "mu_norm", cfg=cfg, title="||mu(z)|| over time")
    H.hist_param(dfp, "mu_norm", cfg=cfg)
    print("Saved mu_norm over-time + hist plots.")


def plot_L_on_date(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    figures_dir=None,
    tag="L_cholesky",
):
    """
    Pulls sigma = out[7] (your Cholesky L) for one curve per currency on date_pick,
    then:
      (1) plots Frobenius norm of L per currency
      (2) plots each entry L[i,j] across currencies
      (3) prints L matrices
    """
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    device = next(model.parameters()).device
    xb = X_tensor_cpu[idxs].to(device)

    model.eval()
    out = model(xb)
    sigma = out[7]  # Cholesky L (B,d,d)

    if sigma.ndim != 3:
        raise ValueError(f"Expected sigma/L to be (B,d,d), got {tuple(sigma.shape)}")

    B, d, _ = sigma.shape
    sig_np = sigma.detach().cpu().numpy()

    # (1) Frobenius norm per currency
    L_frob = np.linalg.norm(sig_np.reshape(B, -1), axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.scatter(i, L_frob[i], color=currency_colors.get(ccy, None), label=ccy)
    ax.set_xticks(range(B))
    ax.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")
    ax.set_ylabel("||L||_F")
    ax.set_title(f"Cholesky L magnitude on {pd.to_datetime(date_pick).date()}")
    ax.grid(True)
    fig.tight_layout()

    if figures_dir is not None:
        path = os.path.join(figures_dir, f"{tag}_frob_{pd.to_datetime(date_pick).date()}.png")
        fig.savefig(path, dpi=250)
        print("Saved:", path)
    else:
        H.save_figure(fig, plot_cfg, f"{tag}_frob_{pd.to_datetime(date_pick).date()}")

    # (2) Entry-wise scatter across currencies
    fig2, axes = plt.subplots(nrows=d, ncols=d, figsize=(10, 7), sharex=True)
    if d == 1:
        axes = np.array([[axes]])

    x = np.arange(B)
    for i in range(d):
        for j in range(d):
            ax2 = axes[i, j]
            vals = sig_np[:, i, j]
            for k, ccy in enumerate(sel_meta["ccy"].values):
                ax2.scatter(x[k], vals[k], color=currency_colors.get(ccy, None), alpha=0.9)
            ax2.set_title(f"L[{i+1},{j+1}]")
            ax2.grid(True)

    for ax2 in axes[-1, :]:
        ax2.set_xticks(x)
        ax2.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")

    fig2.suptitle(f"Entries of Cholesky L on {pd.to_datetime(date_pick).date()}", y=1.02)
    fig2.tight_layout()

    if figures_dir is not None:
        path2 = os.path.join(figures_dir, f"{tag}_entries_{pd.to_datetime(date_pick).date()}.png")
        fig2.savefig(path2, dpi=250)
        print("Saved:", path2)
    else:
        H.save_figure(fig2, plot_cfg, f"{tag}_entries_{pd.to_datetime(date_pick).date()}")

    # (3) Print matrices
    print("\nPer-currency L matrices:")
    for i, ccy in enumerate(sel_meta["ccy"].values):
        print(ccy, "\n", sig_np[i])

    return sigma, sel_meta


def LN_term_decomposition_with_sanity_prints(
    model,
    xb_one: torch.Tensor,          # (1,8)
    tau_max=30,
    print_taus=(5, 10, 30),
):
    """
    LN = -dN/dtau - rN + mu·∇N + 0.5 Tr(Cov Hess N)

    Prints:
      mu, ||mu||, r_tilde, L diag,
      ||∇N|| and mu·∇N at selected maturities.
    Returns terms for plotting.
    """
    device = xb_one.device
    dtype = xb_one.dtype
    model.eval()

    # forward WITH graph
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(xb_one.requires_grad_(True))

    with torch.no_grad():
        mu_np = mu.detach().cpu().numpy()
        print("\n[Sanity] mu:", mu_np)
        print("[Sanity] ||mu||:", float(mu.norm().detach().cpu()))
        print("[Sanity] r_tilde:", float(r_tilde.detach().cpu().view(-1)[0]))
        L_np = sigma.detach().cpu().numpy()[0]
        print("[Sanity] L diag:", np.diag(L_np))

    # dN/dtau via centered differences on P_full
    dP_dtau_full = torch.zeros_like(P_full)
    dP_dtau_full[:, 0]  = (P_full[:, 1] - P_full[:, 0])
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    N_tau   = P_full[:, 1:]          # (1,tau_max)
    dN_dtau = dP_dtau_full[:, 1:]    # (1,tau_max)

    r = r_tilde.view(-1, 1)          # (1,1)
    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(1, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(1, tau_max, device=device, dtype=dtype)

    print_taus = set(int(t) for t in print_taus if 1 <= int(t) <= tau_max)

    for m in range(tau_max):
        Nm = N_tau[:, m]
        g = torch.autograd.grad(Nm.sum(), z, create_graph=True)[0]  # (1,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(1, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

        tau_here = m + 1
        if tau_here in print_taus:
            with torch.no_grad():
                print(f"[Sanity] tau={tau_here:2d} ||grad N||:", float(g.norm().detach().cpu()))
                print(f"[Sanity] tau={tau_here:2d} drift term mu·∇N:", float(drift_term[:, m].detach().cpu()))

    term_dN = (-dN_dtau).detach().cpu().numpy().squeeze(0)
    term_rN = (-(r * N_tau)).detach().cpu().numpy().squeeze(0)
    term_mu = drift_term.detach().cpu().numpy().squeeze(0)
    term_tr = trace_term.detach().cpu().numpy().squeeze(0)
    LN      = term_dN + term_rN + term_mu + term_tr

    return {
        "terms": {
            "minus_dN_dt": term_dN,
            "minus_rN": term_rN,
            "mu_gradN": term_mu,
            "half_trace": term_tr,
            "LN_sum": LN,
        }
    }


def plot_LN_terms_for_one_curve_from_terms(terms_dict, cfg: H.PlotConfig, tag="onecurve", tau_max=30):
    tau_np = np.arange(1, tau_max + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tau_np, terms_dict["minus_dN_dt"], label="-dN/dτ")
    ax.plot(tau_np, terms_dict["minus_rN"], label="-rN")
    ax.plot(tau_np, terms_dict["mu_gradN"], label="μ·∇N")
    ax.plot(tau_np, terms_dict["half_trace"], label="0.5 Tr")
    ax.plot(tau_np, terms_dict["LN_sum"], label="LN (sum)", linewidth=2.5)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Value")
    ax.set_title("LN term decomposition (single curve)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    H.save_figure(fig, cfg, f"LN_terms_{tag}")
    print("Saved LN term decomposition plot.")


def plot_LN_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    LN_np  = LN.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, LN_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("LN residual")
    ax.set_title(f"LN(τ) on {pd.to_datetime(date_pick).date()} (reference)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    H.save_figure(fig, cfg, f"LN_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved LN reference plot.")


def plot_SR_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    SR_np  = SR.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, SR_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Sharpe ratio (approx, reference)")
    ax.set_title(f"SR(τ) on {pd.to_datetime(date_pick).date()} (reference)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    H.save_figure(fig, cfg, f"SR_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved SR reference plot.")


# ============================================================
# RUN THE DIAGNOSTICS
# ============================================================

# 0) mu_norm plots (needs params_df + mu_cols from section 10)
try:
    plot_mu_norm_over_time_and_hist(params_df, mu_cols, plot_cfg)
except Exception as e:
    print("mu_norm plots failed (continuing):", repr(e))

# 1) Choose date used for “one curve per currency” plots
paper_date = pd.to_datetime("2016-08-30")
date_pick = paper_date if (meta_eval["as_of_date"] == paper_date).any() else meta_eval["as_of_date"].iloc[0]

# 2) Pick a finite curve index for single-curve LN decomposition
with torch.no_grad():
    finite_mask = torch.isfinite(X_tensor).all(dim=1)
i0 = int(torch.nonzero(finite_mask, as_tuple=False)[0].item())
xb1 = X_tensor[i0:i0+1].to(device)

print("\nSingle curve index used:", i0)

# 3) Quick reference numbers (must NOT be under no_grad)
model.eval()
N1, LN1, SR1, tau1 = SR_andreasen_reference(model, xb1, tau_max=30, sigma_bar=0.006)
print("SR min/max:", float(SR1.min().detach().cpu()), float(SR1.max().detach().cpu()))
print("N(30Y):",  float(N1[0, -1].detach().cpu()))
print("LN(30Y):", float(LN1[0, -1].detach().cpu()))
print("SR(30Y):", float(SR1[0, -1].detach().cpu()))

# 4) LN decomposition + sanity prints (mu, ||mu||, L diag, ||grad N|| at taus)
decomp = LN_term_decomposition_with_sanity_prints(
    model=model,
    xb_one=xb1,
    tau_max=30,
    print_taus=(5, 10, 30),
)

# 5) Plot decomposition terms (single curve)
plot_LN_terms_for_one_curve_from_terms(
    decomp["terms"],
    cfg=plot_cfg,
    tag=f"idx{i0}",
    tau_max=30,
)

# 6) LN curves across currencies (reference)
plot_LN_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 7) SR curves across currencies (reference)
plot_SR_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 8) Cholesky L magnitude/entries + printed matrices
sigma_L, meta_used = plot_L_on_date(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    figures_dir=FIGURES_DIR,   # set None if you don't want saving here
    tag="L_cholesky",
)

print(f"\nSharpe/LN/L diagnostics saved to: {FIGURES_DIR}")



# ============================================================
# 11) Andreasen Sharpe-ratio diagnostics: N, LN, SR + plots
#    + extra sanity checks (mu norm, grad norm, drift term)
#    + extra plots (mu_norm over time + hist)
#    + L (Cholesky) plots + printed matrices
# ============================================================

from Code.utils.sharpe_ratio import SR_andreasen_reference_noFD


# ---------- helpers ----------
@torch.no_grad()
def pick_one_curve_per_currency_on_date(meta_df: pd.DataFrame, date_pick):
    m = meta_df.copy()
    m["as_of_date"] = pd.to_datetime(m["as_of_date"])
    date_pick = pd.to_datetime(date_pick)

    sel = m[m["as_of_date"] == date_pick].copy()
    if sel.empty:
        raise ValueError(f"No rows in meta_df for date {date_pick.date()}")

    sel = sel.sort_values(["ccy", "as_of_date"]).drop_duplicates(subset=["ccy"], keep="last")
    return sel.index.to_numpy(), sel


def plot_mu_norm_over_time_and_hist(params_df: pd.DataFrame, mu_cols, cfg: H.PlotConfig):
    """
    Adds mu_norm = sqrt(sum_k mu_k^2) then plots over time and histogram.
    Uses your existing helper functions: H.plot_param_over_time, H.hist_param
    """
    mu_cols = [c for c in mu_cols if c in params_df.columns]
    if len(mu_cols) == 0:
        print("No mu columns found for mu_norm plot. Skipping.")
        return

    mu_sq = None
    for c in mu_cols:
        v = params_df[c].astype(float).values
        mu_sq = v * v if mu_sq is None else (mu_sq + v * v)

    dfp = params_df.copy()
    dfp["mu_norm"] = np.sqrt(mu_sq)

    H.plot_param_over_time(dfp, "mu_norm", cfg=cfg, title="||mu(z)|| over time")
    H.hist_param(dfp, "mu_norm", cfg=cfg)
    print("Saved mu_norm over-time + hist plots.")


def plot_L_on_date(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    figures_dir=None,
    tag="L_cholesky",
):
    """
    Pulls sigma = out[7] (your Cholesky L) for one curve per currency on date_pick,
    then:
      (1) plots Frobenius norm of L per currency
      (2) plots each entry L[i,j] across currencies
      (3) prints L matrices
    """
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    device = next(model.parameters()).device
    xb = X_tensor_cpu[idxs].to(device)

    model.eval()
    out = model(xb)
    sigma = out[7]  # Cholesky L (B,d,d)

    if sigma.ndim != 3:
        raise ValueError(f"Expected sigma/L to be (B,d,d), got {tuple(sigma.shape)}")

    B, d, _ = sigma.shape
    sig_np = sigma.detach().cpu().numpy()

    # (1) Frobenius norm per currency
    L_frob = np.linalg.norm(sig_np.reshape(B, -1), axis=1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.scatter(i, L_frob[i], color=currency_colors.get(ccy, None), label=ccy)
    ax.set_xticks(range(B))
    ax.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")
    ax.set_ylabel("||L||_F")
    ax.set_title(f"Cholesky L magnitude on {pd.to_datetime(date_pick).date()}")
    ax.grid(True)
    fig.tight_layout()

    if figures_dir is not None:
        path = os.path.join(figures_dir, f"{tag}_frob_{pd.to_datetime(date_pick).date()}.png")
        fig.savefig(path, dpi=250)
        print("Saved:", path)
    else:
        H.save_figure(fig, plot_cfg, f"{tag}_frob_{pd.to_datetime(date_pick).date()}")

    # (2) Entry-wise scatter across currencies
    fig2, axes = plt.subplots(nrows=d, ncols=d, figsize=(10, 7), sharex=True)
    if d == 1:
        axes = np.array([[axes]])

    x = np.arange(B)
    for i in range(d):
        for j in range(d):
            ax2 = axes[i, j]
            vals = sig_np[:, i, j]
            for k, ccy in enumerate(sel_meta["ccy"].values):
                ax2.scatter(x[k], vals[k], color=currency_colors.get(ccy, None), alpha=0.9)
            ax2.set_title(f"L[{i+1},{j+1}]")
            ax2.grid(True)

    for ax2 in axes[-1, :]:
        ax2.set_xticks(x)
        ax2.set_xticklabels(sel_meta["ccy"].values, rotation=45, ha="right")

    fig2.suptitle(f"Entries of Cholesky L on {pd.to_datetime(date_pick).date()}", y=1.02)
    fig2.tight_layout()

    if figures_dir is not None:
        path2 = os.path.join(figures_dir, f"{tag}_entries_{pd.to_datetime(date_pick).date()}.png")
        fig2.savefig(path2, dpi=250)
        print("Saved:", path2)
    else:
        H.save_figure(fig2, plot_cfg, f"{tag}_entries_{pd.to_datetime(date_pick).date()}")

    # (3) Print matrices
    print("\nPer-currency L matrices:")
    for i, ccy in enumerate(sel_meta["ccy"].values):
        print(ccy, "\n", sig_np[i])

    return sigma, sel_meta


def LN_term_decomposition_with_sanity_prints(
    model,
    xb_one: torch.Tensor,          # (1,8)
    tau_max=30,
    print_taus=(5, 10, 30),
):
    """
    LN = -dN/dtau - rN + mu·∇N + 0.5 Tr(Cov Hess N)

    Prints:
      mu, ||mu||, r_tilde, L diag,
      ||∇N|| and mu·∇N at selected maturities.
    Returns terms for plotting.
    """
    device = xb_one.device
    dtype = xb_one.dtype
    model.eval()

    # forward WITH graph
    S_hat, z, P_full, A_vals, B_vals, G_vals, mu, sigma, r_tilde = model(xb_one.requires_grad_(True))

    with torch.no_grad():
        mu_np = mu.detach().cpu().numpy()
        print("\n[Sanity] mu:", mu_np)
        print("[Sanity] ||mu||:", float(mu.norm().detach().cpu()))
        print("[Sanity] r_tilde:", float(r_tilde.detach().cpu().view(-1)[0]))
        L_np = sigma.detach().cpu().numpy()[0]
        print("[Sanity] L diag:", np.diag(L_np))

    # dN/dtau via centered differences on P_full
    dP_dtau_full = torch.zeros_like(P_full)
    dP_dtau_full[:, 0]  = (P_full[:, 1] - P_full[:, 0])
    dP_dtau_full[:, -1] = (P_full[:, -1] - P_full[:, -2])
    if tau_max >= 2:
        dP_dtau_full[:, 1:-1] = 0.5 * (P_full[:, 2:] - P_full[:, :-2])

    N_tau   = P_full[:, 1:]          # (1,tau_max)
    dN_dtau = dP_dtau_full[:, 1:]    # (1,tau_max)

    r = r_tilde.view(-1, 1)          # (1,1)
    d = z.shape[1]
    sigma_cols = [sigma[:, :, j] for j in range(d)]

    drift_term = torch.zeros(1, tau_max, device=device, dtype=dtype)
    trace_term = torch.zeros(1, tau_max, device=device, dtype=dtype)

    print_taus = set(int(t) for t in print_taus if 1 <= int(t) <= tau_max)

    for m in range(tau_max):
        Nm = N_tau[:, m]
        g = torch.autograd.grad(Nm.sum(), z, create_graph=True)[0]  # (1,d)

        drift_term[:, m] = (g * mu).sum(dim=1)

        hvp_sum = torch.zeros(1, device=device, dtype=dtype)
        for v in sigma_cols:
            gv = (g * v).sum()
            Hg_v = torch.autograd.grad(gv, z, create_graph=True)[0]
            hvp_sum += (Hg_v * v).sum(dim=1)

        trace_term[:, m] = 0.5 * hvp_sum

        tau_here = m + 1
        if tau_here in print_taus:
            with torch.no_grad():
                print(f"[Sanity] tau={tau_here:2d} ||grad N||:", float(g.norm().detach().cpu()))
                print(f"[Sanity] tau={tau_here:2d} drift term mu·∇N:", float(drift_term[:, m].detach().cpu()))

    term_dN = (-dN_dtau).detach().cpu().numpy().squeeze(0)
    term_rN = (-(r * N_tau)).detach().cpu().numpy().squeeze(0)
    term_mu = drift_term.detach().cpu().numpy().squeeze(0)
    term_tr = trace_term.detach().cpu().numpy().squeeze(0)
    LN      = term_dN + term_rN + term_mu + term_tr

    return {
        "terms": {
            "minus_dN_dt": term_dN,
            "minus_rN": term_rN,
            "mu_gradN": term_mu,
            "half_trace": term_tr,
            "LN_sum": LN,
        }
    }


def plot_LN_terms_for_one_curve_from_terms(terms_dict, cfg: H.PlotConfig, tag="onecurve", tau_max=30):
    tau_np = np.arange(1, tau_max + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tau_np, terms_dict["minus_dN_dt"], label="-dN/dτ")
    ax.plot(tau_np, terms_dict["minus_rN"], label="-rN")
    ax.plot(tau_np, terms_dict["mu_gradN"], label="μ·∇N")
    ax.plot(tau_np, terms_dict["half_trace"], label="0.5 Tr")
    ax.plot(tau_np, terms_dict["LN_sum"], label="LN (sum)", linewidth=2.5)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Value")
    ax.set_title("LN term decomposition (single curve)")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    H.save_figure(fig, cfg, f"LN_terms_{tag}")
    print("Saved LN term decomposition plot.")


def plot_LN_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference_noFD(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    LN_np  = LN.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, LN_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("LN residual")
    ax.set_title(f"LN(τ) on {pd.to_datetime(date_pick).date()} (reference)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    H.save_figure(fig, cfg, f"LN_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved LN reference plot.")


def plot_SR_on_date_reference(
    model,
    X_tensor_cpu: torch.Tensor,
    meta_df: pd.DataFrame,
    currency_colors: dict,
    date_pick,
    cfg: H.PlotConfig,
    tau_max=30,
    sigma_bar=0.006,
):
    idxs, sel_meta = pick_one_curve_per_currency_on_date(meta_df, date_pick)
    xb = X_tensor_cpu[idxs].to(next(model.parameters()).device)

    model.eval()
    N_tau, LN, SR, tau = SR_andreasen_reference_noFD(model, xb, tau_max=tau_max, sigma_bar=sigma_bar)

    tau_np = tau.detach().cpu().numpy()
    SR_np  = SR.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, ccy in enumerate(sel_meta["ccy"].values):
        ax.plot(tau_np, SR_np[i], label=ccy, color=currency_colors.get(ccy, None), alpha=0.9)

    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Sharpe ratio (approx, reference)")
    ax.set_title(f"SR(τ) on {pd.to_datetime(date_pick).date()} (reference)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9)
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    H.save_figure(fig, cfg, f"SR_reference_{pd.to_datetime(date_pick).date()}")
    print("Saved SR reference plot.")


# ============================================================
# RUN THE DIAGNOSTICS
# ============================================================

# 0) mu_norm plots (needs params_df + mu_cols from section 10)
try:
    plot_mu_norm_over_time_and_hist(params_df, mu_cols, plot_cfg)
except Exception as e:
    print("mu_norm plots failed (continuing):", repr(e))

# 1) Choose date used for “one curve per currency” plots
paper_date = pd.to_datetime("2016-08-30")
date_pick = paper_date if (meta_eval["as_of_date"] == paper_date).any() else meta_eval["as_of_date"].iloc[0]

# 2) Pick a finite curve index for single-curve LN decomposition
with torch.no_grad():
    finite_mask = torch.isfinite(X_tensor).all(dim=1)
i0 = int(torch.nonzero(finite_mask, as_tuple=False)[0].item())
xb1 = X_tensor[i0:i0+1].to(device)

print("\nSingle curve index used:", i0)

# 3) Quick reference numbers (must NOT be under no_grad)
model.eval()
N1, LN1, SR1, tau1 = SR_andreasen_reference_noFD(model, xb1, tau_max=30, sigma_bar=0.006)
print("SR min/max:", float(SR1.min().detach().cpu()), float(SR1.max().detach().cpu()))
print("N(30Y):",  float(N1[0, -1].detach().cpu()))
print("LN(30Y):", float(LN1[0, -1].detach().cpu()))
print("SR(30Y):", float(SR1[0, -1].detach().cpu()))

# 4) LN decomposition + sanity prints (mu, ||mu||, L diag, ||grad N|| at taus)
decomp = LN_term_decomposition_with_sanity_prints(
    model=model,
    xb_one=xb1,
    tau_max=30,
    print_taus=(5, 10, 30),
)

# 5) Plot decomposition terms (single curve)
plot_LN_terms_for_one_curve_from_terms(
    decomp["terms"],
    cfg=plot_cfg,
    tag=f"idx{i0}",
    tau_max=30,
)

# 6) LN curves across currencies (reference)
plot_LN_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 7) SR curves across currencies (reference)
plot_SR_on_date_reference(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    cfg=plot_cfg,
    tau_max=30,
    sigma_bar=0.006,
)

# 8) Cholesky L magnitude/entries + printed matrices
sigma_L, meta_used = plot_L_on_date(
    model=model,
    X_tensor_cpu=X_tensor,
    meta_df=meta_eval,
    currency_colors=currency_color_map,
    date_pick=date_pick,
    figures_dir=FIGURES_DIR,   # set None if you don't want saving here
    tag="L_cholesky",
)

print(f"\nSharpe/LN/L diagnostics saved to: {FIGURES_DIR}")