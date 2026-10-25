# -*- coding: utf-8 -*-
"""相関・回帰分析アプリ（改良版）

必要パッケージ: streamlit, pandas, numpy, scipy, statsmodels, plotly
  pip install streamlit pandas numpy scipy statsmodels plotly
起動: streamlit run correlation_regression_app.py
"""
import inspect
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

st.set_page_config(page_title="相関・回帰分析アプリ", layout="wide")
st.title("📊 相関・回帰分析アプリ")

INTERCEPT = "切片 (Intercept)"
_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


# ==========================================
# ユーティリティ
# ==========================================
def show_plot(fig):
    """Streamlitのバージョン差（use_container_width廃止）を吸収"""
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(show_spinner=False)
def load_csv(raw: bytes):
    """UTF-8(BOM可) → CP932(Shift-JIS) の順で読み込む"""
    last_err = None
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc), enc
        except UnicodeDecodeError as e:
            last_err = e
    raise ValueError(f"文字コードを判別できませんでした: {last_err}")


def parse_pasted(text: str) -> pd.DataFrame:
    """タブ → カンマ → 空白 の順で区切り文字を判定"""
    text = text.strip()
    first = text.splitlines()[0]
    sep = "\t" if "\t" in first else ("," if "," in first else r"\s+")
    return pd.read_csv(io.StringIO(text), sep=sep)


def make_unique(cols):
    seen, out = {}, []
    for c in cols:
        c = str(c).strip()
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def coerce_numeric(d: pd.DataFrame):
    """「1,234」のような桁区切り文字列を数値化（90%以上が数値化できる列のみ）"""
    converted = []
    for c in d.columns:
        s = d[c]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            n_valid = s.notna().sum()
            if n_valid == 0:
                continue
            cleaned = pd.to_numeric(
                s.astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            if cleaned.notna().sum() >= 0.9 * n_valid:
                d[c] = cleaned
                converted.append(c)
    return d, converted


def stars(p):
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def tex_escape(s: str) -> str:
    table = {
        "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_",
        "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "^": r"\^{}", "~": r"\~{}",
    }
    return "".join(table.get(ch, ch) for ch in s)


def corr_tables(d: pd.DataFrame, method: str):
    """ペアワイズ欠損除去で r, p, n を計算"""
    cols = list(d.columns)
    k = len(cols)
    r = np.full((k, k), np.nan)
    p = np.full((k, k), np.nan)
    n = np.zeros((k, k), dtype=int)
    func = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}[method]
    for i in range(k):
        for j in range(i, k):
            pair = d[[cols[i], cols[j]]].dropna()
            n[i, j] = n[j, i] = len(pair)
            if i == j:
                r[i, j], p[i, j] = 1.0, 0.0
                continue
            if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                res = func(pair.iloc[:, 0], pair.iloc[:, 1])
                r[i, j] = r[j, i] = res[0]
                p[i, j] = p[j, i] = res[1]
    mk = lambda a: pd.DataFrame(a, index=cols, columns=cols)
    return mk(r), mk(p), mk(n)


def adjust_p(p: pd.DataFrame, how: str) -> pd.DataFrame:
    if how == "なし":
        return p
    iu = np.triu_indices(len(p), k=1)
    vals = p.values[iu].astype(float)
    mask = ~np.isnan(vals)
    adj = vals.copy()
    if mask.any():
        adj[mask] = multipletests(vals[mask], method={"Holm": "holm", "Bonferroni": "bonferroni"}[how])[1]
    out = np.array(p.values, dtype=float)
    out[iu] = adj
    out[(iu[1], iu[0])] = adj
    return pd.DataFrame(out, index=p.index, columns=p.columns)


# ==========================================
# 1. データの読み込み
# ==========================================
st.header("1. データの読み込み")
source = st.radio(
    "入力方法（どちらか一方のみ使用されます）",
    ["CSVファイル", "コピー＆ペースト（Excel等）"],
    horizontal=True,
)

df = None
if source == "CSVファイル":
    uploaded = st.file_uploader("CSVファイルを選択してください（UTF-8 / Shift-JIS 自動判別）", type=["csv"])
    if uploaded is not None:
        try:
            df, enc = load_csv(uploaded.getvalue())
            st.caption(f"文字コード: {enc}")
        except Exception as e:
            st.error(f"ファイルの読み込みに失敗しました: {e}")
else:
    pasted = st.text_area("1行目はヘッダー行にしてください（タブ・カンマ・空白区切りに対応）", height=200)
    if pasted.strip():
        try:
            df = parse_pasted(pasted)
        except Exception as e:
            st.error(f"データの読み込みに失敗しました: {e}")

if df is None:
    st.info("データを読み込んでください。")
    st.stop()

df = df.copy()
df.columns = make_unique(df.columns)
df, converted_cols = coerce_numeric(df)
if converted_cols:
    st.info(f"桁区切り等を除去して数値化した列: {', '.join(converted_cols)}（意図しない変換でないか確認してください）")

st.subheader("読み込んだデータ（先頭5行）")
st.dataframe(df.head())
st.caption(f"{df.shape[0]} 行 × {df.shape[1]} 列")

num_all = df.select_dtypes(include="number")
const_cols = [c for c in num_all.columns if num_all[c].nunique(dropna=True) <= 1]
usable = [c for c in num_all.columns if c not in const_cols and num_all[c].notna().sum() >= 3]
if const_cols:
    st.warning(f"定数列（値が1種類以下）は分析から除外しました: {', '.join(const_cols)}")

if len(usable) < 2:
    st.warning("⚠️ 分析には、有効な数値列（定数でなく3件以上の値）が2つ以上必要です。")
    st.stop()

sel = st.multiselect("分析対象の数値変数", usable, default=usable)
if len(sel) < 2:
    st.warning("2つ以上の変数を選択してください。")
    st.stop()
num = df[sel]

with st.expander("基本統計量・欠損数"):
    desc = num.describe().T
    desc["欠損数"] = num.isna().sum()
    st.dataframe(desc.style.format("{:.4g}", na_rep="-"))

st.divider()

# ==========================================
# 2. 相関分析
# ==========================================
st.header("2. 相関分析")

c_a, c_b = st.columns(2)
with c_a:
    method_label = st.selectbox(
        "相関係数の計算手法",
        (
            "ピアソン (Pearson) - 線形関係・外れ値に敏感",
            "スピアマン (Spearman) - 順位相関・単調関係",
            "ケンドール (Kendall) - 順位相関・小標本/同順位が多い場合",
        ),
    )
with c_b:
    adj_label = st.selectbox(
        "多重比較補正（全ペアを同時に検定するため）",
        ["Holm", "Bonferroni", "なし"],
        help="変数が k 個なら k(k-1)/2 回の検定。補正なしだと偽陽性が増えます。",
    )
method = "pearson" if "ピアソン" in method_label else "spearman" if "スピアマン" in method_label else "kendall"

r_mat, p_mat, n_mat = corr_tables(num, method)
p_adj = adjust_p(p_mat, adj_label)

cols = list(r_mat.columns)
text = [
    [
        "" if np.isnan(r_mat.iat[i, j]) else f"{r_mat.iat[i, j]:.2f}" + ("" if i == j else stars(p_adj.iat[i, j]))
        for j in range(len(cols))
    ]
    for i in range(len(cols))
]
fig_corr = go.Figure(
    go.Heatmap(
        z=r_mat.values, x=cols, y=cols, zmin=-1, zmax=1, colorscale="RdBu_r",
        text=text, texttemplate="%{text}", hoverongaps=False,
        colorbar=dict(title="r"),
    )
)
fig_corr.update_yaxes(autorange="reversed")
fig_corr.update_layout(height=max(420, 70 * len(cols)), margin=dict(l=10, r=10, t=30, b=10))
st.caption(f"星印は補正後p値（{adj_label}）: * <0.05, ** <0.01, *** <0.001。欠損はペアごとに除外。")
show_plot(fig_corr)

t_r, t_p, t_n = st.tabs(["相関係数 r", f"p値（{adj_label}補正後）", "ペア別 n"])
with t_r:
    st.dataframe(r_mat.style.format("{:.3f}", na_rep="-"))
with t_p:
    st.dataframe(p_adj.style.format("{:.4g}", na_rep="-"))
with t_n:
    st.dataframe(n_mat)

st.download_button(
    "相関係数行列をCSVで保存",
    r_mat.to_csv().encode("utf-8-sig"),
    file_name=f"correlation_{method}.csv",
    mime="text/csv",
)

with st.expander("ペアを選んで散布図を確認（相関係数だけで判断しないために）"):
    pc1, pc2 = st.columns(2)
    pa = pc1.selectbox("変数A", sel, index=0, key="pair_a")
    pb = pc2.selectbox("変数B", sel, index=min(1, len(sel) - 1), key="pair_b")
    if pa == pb:
        st.info("異なる2変数を選択してください。")
    else:
        pair = num[[pa, pb]].dropna()
        if len(pair) >= 3 and pair[pa].nunique() > 1 and pair[pb].nunique() > 1:
            rp, pp = stats.pearsonr(pair[pa], pair[pb])
            rs, ps = stats.spearmanr(pair[pa], pair[pb])
            st.markdown(
                f"n = {len(pair)} ／ Pearson r = **{rp:.3f}** (p={pp:.3g}) ／ Spearman ρ = **{rs:.3f}** (p={ps:.3g})"
            )
            if abs(rp - rs) > 0.2:
                st.warning("PearsonとSpearmanの差が大きい → 外れ値・非線形性・影響点の可能性。")
            fig_pair = px.scatter(pair, x=pa, y=pb, trendline="ols", trendline_color_override="red", opacity=0.75)
            show_plot(fig_pair)
        else:
            st.warning("有効データが不足しています。")

st.divider()

# ==========================================
# 3. 回帰分析
# ==========================================
st.header("3. 回帰分析（単回帰・重回帰）")

c3, c4 = st.columns(2)
with c3:
    y_var = st.selectbox("目的変数 (Y): 結果となる変数", sel, index=min(1, len(sel) - 1))
with c4:
    x_options = [c for c in sel if c != y_var]
    x_vars = st.multiselect("説明変数 (X): 複数選択可（Yは自動で除外）", x_options, default=x_options[:1])

robust = st.checkbox(
    "頑健標準誤差 (HC3) を使用",
    value=False,
    help="分散が不均一（残差の広がりが一定でない）疑いがある場合の対処。係数の推定値は変わらず、SE・p値・信頼区間が変わります。",
)

if not x_vars:
    st.info("説明変数を1つ以上選択してください。")
    st.stop()

reg_df = num[x_vars + [y_var]].dropna()
n_obs, k_x = len(reg_df), len(x_vars)
if n_obs < len(num):
    st.caption(f"欠損のため {len(num) - n_obs} 行を除外 → 使用 n = {n_obs}")

if n_obs < k_x + 2:
    st.error("有効なデータ数が不足しているため、回帰分析を実行できません。")
    st.stop()
if any(reg_df[c].nunique() <= 1 for c in x_vars + [y_var]):
    st.error("欠損除外後に定数となる変数があります。変数の選択を見直してください。")
    st.stop()

X = reg_df[x_vars].copy()
X.insert(0, INTERCEPT, 1.0)
y = reg_df[y_var]

if np.linalg.matrix_rank(X.values) < X.shape[1]:
    st.error("説明変数間に完全な多重共線性（線形従属）があります。いずれかの変数を除外してください。")
    st.stop()

ols = sm.OLS(y, X).fit()  # 診断・予測帯用（通常のOLS）
model = sm.OLS(y, X).fit(cov_type="HC3", use_t=True) if robust else ols

if n_obs / k_x < 10:
    st.warning(f"説明変数1つあたりのデータ数が {n_obs / k_x:.1f} 件です（目安: 10件以上）。過学習・不安定な推定に注意。")

# ---- 回帰式 ----
st.subheader("回帰式（推定式）")
eq_parts = [f"{model.params[INTERCEPT]:.4g}"]
plain_parts = [f"{model.params[INTERCEPT]:.4g}"]
for v in x_vars:
    c = model.params[v]
    sign = "+" if c >= 0 else "-"
    eq_parts.append(f"{sign} {abs(c):.4g} \\times \\text{{{tex_escape(v)}}}")
    plain_parts.append(f"{sign} {abs(c):.4g} × {v}")
st.latex(f"\\hat{{\\text{{{tex_escape(y_var)}}}}} = " + " ".join(eq_parts))
st.code(f"{y_var}(予測) = " + " ".join(plain_parts), language=None)

# ---- 指標 ----
st.subheader("分析結果サマリー")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("n", f"{n_obs}")
m2.metric("決定係数 R²", f"{model.rsquared:.4f}", help="Xで説明できるYの変動割合。因果や予測精度の保証ではありません。")
m3.metric("調整済 R²", f"{model.rsquared_adj:.4f}", help="説明変数の数で補正。変数が多い場合はこちらを参照。")
m4.metric("F検定 p値", f"{model.f_pvalue:.3e}", help="全係数が0という帰無仮説の検定。")
m5.metric("残差標準誤差", f"{np.sqrt(ols.mse_resid):.4g}", help="Yと同じ単位での予測誤差の目安。")

# ---- 係数表 ----
t_coef, t_fit, t_diag = st.tabs(["偏回帰係数", "当てはまり", "残差診断"])

with t_coef:
    ci = model.conf_int()
    coef_df = pd.DataFrame(
        {
            "偏回帰係数": model.params,
            "標準誤差" + (" (HC3)" if robust else ""): model.bse,
            "t値": model.tvalues,
            "p値": model.pvalues,
            "95%CI 下限": ci[0],
            "95%CI 上限": ci[1],
        }
    )
    sd_y = y.std()
    coef_df["標準化係数 β"] = [np.nan] + [model.params[v] * reg_df[v].std() / sd_y for v in x_vars]
    if k_x >= 2:
        vifs = [variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])]
        coef_df["VIF"] = [np.nan] + vifs
        if np.nanmax(vifs) > 10:
            st.warning("VIF > 10 の変数があります。多重共線性により係数が不安定です（相関の高いXの片方を除外、等）。")
        elif np.nanmax(vifs) > 5:
            st.info("VIF > 5 の変数があります。係数の解釈に注意してください。")
    st.dataframe(coef_df.style.format("{:.4g}", na_rep="-"))
    st.caption("標準化係数 β: 各Xが1SD増えたときのYの変化（SD単位）。単位の異なるXの影響度の比較用。")
    st.download_button(
        "係数表をCSVで保存",
        coef_df.to_csv().encode("utf-8-sig"),
        file_name="regression_coefficients.csv",
        mime="text/csv",
    )

with t_fit:
    if k_x == 1:
        xv = x_vars[0]
        grid = np.linspace(reg_df[xv].min(), reg_df[xv].max(), 100)
        gx = pd.DataFrame({INTERCEPT: 1.0, xv: grid})
        sf = ols.get_prediction(gx).summary_frame(alpha=0.05)
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=np.r_[grid, grid[::-1]], y=np.r_[sf["obs_ci_upper"], sf["obs_ci_lower"][::-1]],
            fill="toself", fillcolor="rgba(150,150,150,0.15)", line=dict(width=0), name="95%予測区間", hoverinfo="skip"))
        fig_fit.add_trace(go.Scatter(
            x=np.r_[grid, grid[::-1]], y=np.r_[sf["mean_ci_upper"], sf["mean_ci_lower"][::-1]],
            fill="toself", fillcolor="rgba(255,0,0,0.2)", line=dict(width=0), name="95%信頼区間（回帰直線）", hoverinfo="skip"))
        fig_fit.add_trace(go.Scatter(x=grid, y=sf["mean"], mode="lines", line=dict(color="red"), name="回帰直線"))
        fig_fit.add_trace(go.Scatter(x=reg_df[xv], y=y, mode="markers", opacity=0.75, name="実測値"))
        fig_fit.update_layout(xaxis_title=xv, yaxis_title=y_var, title=f"{y_var} ~ {xv}")
        show_plot(fig_fit)
    else:
        pred = ols.fittedvalues
        lo, hi = float(min(pred.min(), y.min())), float(max(pred.max(), y.max()))
        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(x=pred, y=y, mode="markers", opacity=0.75, name="データ",
                                     text=[f"行 {i}" for i in reg_df.index]))
        fig_fit.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="red", dash="dash"),
                                     name="予測値＝実測値"))
        fig_fit.update_layout(xaxis_title="予測値", yaxis_title=f"実測値 ({y_var})",
                              title=f"{y_var}の実測値と予測値")
        show_plot(fig_fit)
    st.caption("※ 同一データで当てはめた結果です（学習データ内の精度）。汎化性能は交差検証や独立データで確認が必要です。")

with t_diag:
    resid = ols.resid
    infl = ols.get_influence()
    cooks = infl.cooks_distance[0]

    d1, d2 = st.columns(2)
    with d1:
        fig_rf = px.scatter(x=ols.fittedvalues, y=resid, labels={"x": "予測値", "y": "残差"},
                            title="残差 vs 予測値（ランダムな帯状が理想）")
        fig_rf.add_hline(y=0, line_color="red", line_dash="dash")
        show_plot(fig_rf)
    with d2:
        (osm, osr), (slope, icpt, _) = stats.probplot(resid, dist="norm")
        fig_qq = go.Figure()
        fig_qq.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="残差"))
        fig_qq.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[slope * osm.min() + icpt, slope * osm.max() + icpt],
                                    mode="lines", line=dict(color="red"), name="基準線"))
        fig_qq.update_layout(title="正規Q-Qプロット（直線上が理想）", xaxis_title="理論分位点", yaxis_title="残差の分位点",
                             showlegend=False)
        show_plot(fig_qq)

    thr = 4 / n_obs
    fig_cd = px.bar(x=[str(i) for i in reg_df.index], y=cooks, labels={"x": "行", "y": "Cook's D"},
                    title=f"Cook's distance（目安 4/n = {thr:.3f}）")
    fig_cd.add_hline(y=thr, line_color="red", line_dash="dash")
    show_plot(fig_cd)

    flagged = np.where(cooks > thr)[0]
    if len(flagged):
        st.markdown("**影響の大きい観測値（Cook's D > 4/n）**")
        st.dataframe(pd.DataFrame({
            "行": reg_df.index[flagged],
            "Cook's D": cooks[flagged],
            "てこ比": infl.hat_matrix_diag[flagged],
            "外れ値度 (外部スチューデント化残差)": infl.resid_studentized_external[flagged],
        }).style.format({"Cook's D": "{:.4f}", "てこ比": "{:.4f}", "外れ値度 (外部スチューデント化残差)": "{:.3f}"}),
            hide_index=True)
        st.caption("該当行を除いて再分析し、結論が変わるか確認してください（除外の根拠は別途必要）。")

    rows = []
    if 3 <= n_obs <= 5000:
        sh = stats.shapiro(resid)
        rows.append(("Shapiro-Wilk（残差の正規性）", sh.statistic, sh.pvalue,
                     "正規性に疑い" if sh.pvalue < 0.05 else "有意な逸脱なし"))
    bp = het_breuschpagan(resid, ols.model.exog)
    rows.append(("Breusch-Pagan（等分散性）", bp[0], bp[1],
                 "不均一分散の疑い → HC3推奨" if bp[1] < 0.05 else "有意な逸脱なし"))
    dw = durbin_watson(resid)
    rows.append(("Durbin-Watson（自己相関）", dw, np.nan,
                 "自己相関の疑い" if (dw < 1.5 or dw > 2.5) else "2付近（問題小）"))
    st.dataframe(pd.DataFrame(rows, columns=["検定", "統計量", "p値", "判定"]).style.format(
        {"統計量": "{:.4g}", "p値": "{:.4g}"}, na_rep="-"), hide_index=True)
    st.caption("検定は小標本では検出力不足、大標本では過敏です。p値だけでなく上の図で判断してください。"
               "Durbin-Watsonは行の並びが時間・圃場順などの意味を持つ場合のみ有効です。")

with st.expander("解釈上の注意"):
    st.markdown(
        "- 相関・回帰は**関連**を示すもので、因果ではない（交絡・逆因果の可能性）。\n"
        "- 相関分析で有意だった変数を回帰に入れる手順は、検定の繰り返しで偽陽性を増やす。**変数は事前の仮説で決める**のが原則。\n"
        "- 反復・ブロック・年次などの**非独立データ**（同一圃場の反復測定等）には本アプリのOLSは不適。混合モデル等が必要。\n"
        "- 外挿（Xの観測範囲外の予測）は信頼できない。"
    )