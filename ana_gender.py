# -*- coding: utf-8 -*-
"""
Gender Diversity (%) vs Remote Work Ratio (%) — FIXED VERSION
- Fix: sanitize column names before statsmodels formula
- Fix: silence pandas GroupBy.apply FutureWarning
- Plots & report saved under {csv_dir}/figures_gender_remote and {csv_dir}/reports
Requires: pandas, numpy, matplotlib, statsmodels
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from textwrap import dedent

# -----------------------------
# 1) Config
# -----------------------------
CSV_PATH = r"D:\datasci_BIG1\ai_job_trends_dataset_adjusted.csv"
MIN_INDUSTRY_N = 10
REMOTE_BINS = 3
SEED = 42
np.random.seed(SEED)

# -----------------------------
# 2) Helpers
# -----------------------------
def find_col(df, patterns):
    cols = [c.strip() for c in df.columns]
    for pat in patterns:
        regex = re.compile(pat, flags=re.IGNORECASE)
        for c in cols:
            if regex.search(c):
                return c
    return None

def winsorize_series(s, lower=0.01, upper=0.99):
    if s.isna().all():
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)

def standardize(s):
    return (s - s.mean()) / s.std(ddof=0)

def weighted_corr(x, y, w):
    dfw = pd.DataFrame({"x": x, "y": y, "w": w}).dropna()
    if len(dfw) < 2:
        return np.nan
    w = dfw["w"].to_numpy(dtype=float)
    x = dfw["x"].to_numpy(dtype=float)
    y = dfw["y"].to_numpy(dtype=float)
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov_xy = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return np.nan
    return cov_xy / np.sqrt(vx * vy)

def sanitize_colname(name: str) -> str:
    """
    Turn 'Gender Diversity (%)' -> 'gender_diversity_pct'
    Creates a safe Python identifier for patsy/statsmodels formulas.
    """
    s = name.strip().lower()
    # common replacements
    s = s.replace("%", "pct").replace("$", "usd").replace("£", "gbp")
    s = re.sub(r"\(.*?\)", "", s)               # remove parenthetical parts
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s)        # non-alnum -> underscore
    s = re.sub(r"_+", "_", s).strip("_")
    if re.match(r"^\d", s):
        s = "c_" + s
    return s

# -----------------------------
# 3) Load
# -----------------------------
try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
df.columns = [c.strip() for c in df.columns]

# Identify original column names
col_remote = find_col(df, [r"remote\s*work\s*ratio", r"\bremote\b"])
col_gender = find_col(df, [r"gender\s*diversity", r"\bdiversity\b"])
col_industry = find_col(df, [r"\bindustry\b"])
col_edu = find_col(df, [r"required\s*education", r"\beducation\b"])
col_exp = find_col(df, [r"experience\s*required", r"\bexperience\b"])
col_salary = find_col(df, [r"median\s*salary", r"\bsalary\b"])
col_risk = find_col(df, [r"automation\s*risk", r"\brisk\b"])
col_openings = find_col(df, [r"job\s*openings\s*\(2024\)|job\s*openings", r"\bopenings\b"])

needed = [col_remote, col_gender, col_industry, col_edu, col_exp, col_salary, col_risk]
missing = [n for n in needed if n is None]
if missing:
    raise ValueError(f"未找到必要列（请检查 CSV 标题）: {missing}")

# numeric conversion on original df (for plotting labels keep originals)
for c in [col_remote, col_gender, col_exp, col_salary, col_risk, col_openings]:
    if c is not None:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -----------------------------
# 4) Build a SAFE-NAME dataframe for modeling
# -----------------------------
# Build mapping original -> safe
name_map = {c: sanitize_colname(c) for c in df.columns}
df_safe = df.rename(columns=name_map).copy()

# resolve safe names
s_remote   = name_map[col_remote]
s_gender   = name_map[col_gender]
s_industry = name_map[col_industry]
s_edu      = name_map[col_edu]
s_exp      = name_map[col_exp]
s_salary   = name_map[col_salary]
s_risk     = name_map[col_risk]

# -----------------------------
# 5) Clean / Features
# -----------------------------
df_safe[f"{s_salary}_clipped"] = winsorize_series(df_safe[s_salary])
df_safe[f"{s_exp}_clipped"] = winsorize_series(df_safe[s_exp])

# Remote quantile buckets (for plots)
valid_remote = df_safe[s_remote].dropna()
if valid_remote.empty:
    raise ValueError("远程比例列全为空，无法分析。")
quantiles = valid_remote.quantile([0, 1/REMOTE_BINS, 2/REMOTE_BINS, 1]).values
labels = [f"Q{i+1}" for i in range(REMOTE_BINS)]
df_safe["remote_bucket"] = pd.cut(df_safe[s_remote], bins=quantiles, labels=labels,
                                  include_lowest=True, duplicates="drop")

# -----------------------------
# 6) Correlations (overall + per-industry + weighted)
# -----------------------------
base = df_safe[[s_remote, s_gender]].dropna()
pearson_all = base[s_remote].corr(base[s_gender], method="pearson")
spearman_all = base[s_remote].corr(base[s_gender], method="spearman")

w_corr = np.nan
if col_openings is not None:
    w_corr = weighted_corr(df_safe[s_remote], df_safe[s_gender], df[col_openings])

# per-industry Spearman; silence FutureWarning via group_keys=False
by_ind = (
    df_safe[[s_industry, s_remote, s_gender]]
    .dropna()
    .groupby(s_industry, group_keys=False)
    .apply(lambda g: pd.Series({
        "n": len(g),
        "spearman": g[s_remote].corr(g[s_gender], method="spearman")
    }))
    .reset_index()
)
by_ind = by_ind[by_ind["n"] >= MIN_INDUSTRY_N].sort_values("spearman", ascending=False)

# -----------------------------
# 7) OLS with controls (using SAFE names, no quotes/backticks)
# -----------------------------
df_safe["_remote_std"] = standardize(df_safe[s_remote])
df_safe["_salary_log"] = np.log1p(df_safe[f"{s_salary}_clipped"])
df_safe["_salary_log_std"] = standardize(df_safe["_salary_log"])
df_safe["_exp_std"] = standardize(df_safe[f"{s_exp}_clipped"])
df_safe["_risk_std"] = standardize(df_safe[s_risk])

reg_cols = [s_gender, "_remote_std", "_salary_log_std", "_exp_std", "_risk_std", s_industry, s_edu]
reg_df = df_safe[reg_cols].dropna()

formula = f"{s_gender} ~ _remote_std + _salary_log_std + _exp_std + _risk_std + C({s_industry}) + C({s_edu})"
model = smf.ols(formula=formula, data=reg_df).fit(cov_type="HC3")

# -----------------------------
# 8) Output dirs
# -----------------------------
base_dir = os.path.dirname(CSV_PATH)
fig_dir = os.path.join(base_dir, "figures_gender_remote")
rep_dir = os.path.join(base_dir, "reports")
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(rep_dir, exist_ok=True)

# -----------------------------
# 9) Plots >= 3
# -----------------------------
# (1) Scatter + OLS line (overall)
x = base[s_remote].to_numpy()
y = base[s_gender].to_numpy()
slope, intercept = np.polyfit(x, y, 1)
xs = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 200)
ys = slope * xs + intercept

plt.figure()
plt.scatter(x, y, alpha=0.5)
plt.plot(xs, ys)
plt.xlabel(col_remote)           # show original label for readability
plt.ylabel(col_gender)
plt.title("Gender Diversity vs Remote Work Ratio (overall)")
plt.grid(True, linestyle="--", alpha=0.3)
plt.figtext(0.01, -0.08, f"Pearson={pearson_all:.3f} | Spearman={spearman_all:.3f} | n={len(base)}", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "scatter_remote_gender.png"), dpi=150)
plt.close()

# (2) Boxplot by remote quantile buckets
buckets = df_safe[["remote_bucket", s_gender]].dropna()
order = [lab for lab in labels if lab in buckets["remote_bucket"].unique().tolist()]
data_for_box = [buckets.loc[buckets["remote_bucket"] == lab, s_gender].values for lab in order]
plt.figure()
plt.boxplot(data_for_box, labels=order, showmeans=True)
plt.xlabel("Remote Work Ratio (quantile buckets)")
plt.ylabel(col_gender)
plt.title("Gender Diversity by Remote Ratio Bucket")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "box_gender_by_remote_bucket.png"), dpi=150)
plt.close()

# (3) By-industry Spearman barh
if not by_ind.empty:
    plt.figure(figsize=(10, max(4, 0.3 * len(by_ind))))
    ylabels = by_ind[s_industry].tolist()
    vals = by_ind["spearman"].values
    ypos = np.arange(len(by_ind))
    plt.barh(ypos, vals)
    plt.yticks(ypos, ylabels)
    plt.axvline(0, color="k", linewidth=0.8)
    plt.xlabel("Spearman correlation (Remote vs Gender Diversity)")
    plt.title(f"By-Industry Spearman (n≥{MIN_INDUSTRY_N})")
    plt.grid(True, axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "industry_spearman_barh.png"), dpi=150)
    plt.close()

# -----------------------------
# 10) Save report + mapping
# -----------------------------
summary_txt = dedent(f"""
    === Gender Diversity vs Remote Work Ratio — Summary ===

    CSV: {CSV_PATH}
    Observations (overall): {len(base)}

    Overall correlations:
      - Pearson:  {pearson_all:.4f}
      - Spearman: {spearman_all:.4f}
      - Weighted Pearson (by '{col_openings}') if available: {w_corr if not np.isnan(w_corr) else 'N/A'}

    OLS with controls (HC3 robust SE):
      Formula (safe names):
        {formula}
      N (regression): {int(model.nobs)}
      R-squared: {model.rsquared:.4f}

      Key coefficient (standardized):
        _remote_std: coef={model.params.get('_remote_std', np.nan):.4f}, 
                     p={model.pvalues.get('_remote_std', np.nan):.4g}

    Notes:
      * Continuous covariates standardized; effect of remote is interpretable per-STD.
      * Industry chart shows groups with n ≥ {MIN_INDUSTRY_N}.
      * Correlation/conditional correlation only; not causal.
      
    Column name mapping (original -> safe):
      { {k: v for k, v in name_map.items() if k in [col_remote, col_gender, col_industry, col_edu, col_exp, col_salary, col_risk]} }
""").strip()

rep_path = os.path.join(rep_dir, "gender_remote_summary.txt")
with open(rep_path, "w", encoding="utf-8") as f:
    f.write(summary_txt + "\n\n" + model.summary().as_text())

print(summary_txt)
print("\n--- Regression full summary saved to:", rep_path)
print("Figures saved to:", fig_dir)
