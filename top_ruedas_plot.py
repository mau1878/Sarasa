"""
top_ruedas_plot.py
==================
Top N de ruedas (días de operación) de UN ticker según una métrica a elegir:

  volumen            Volumen (acciones)
  volumen_nominal    Volumen nominal (precio de cierre × volumen)
  volumen_relativo   Volumen relativo (veces el promedio de las 20 ruedas previas)
  rango              Rango máximo–mínimo (% sobre el mínimo)
  var_cierre_ant     Variación del cierre vs cierre anterior (%)
  apertura_cierre    Distancia apertura → cierre (%)
  gap                Gap de apertura vs cierre anterior (%)

Las métricas con signo (variaciones, apertura→cierre, gap) admiten tres modos:
  "abs"   mayor magnitud (sube o baja)      "up"  mayores subas      "down"  mayores bajas

Uso por consola:
    python top_ruedas_plot.py --ticker AAPL --metric volumen --n 10 --period 1Y
    python top_ruedas_plot.py --ticker NVDA --metric var_cierre_ant --mode down --period 5Y
"""
import argparse
import os
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mtaurus_style  # noqa: F401  (tipografía compartida)
from merval_msci import yf_download
from mtaurus_style import BG_COLOR, PAPER_COLOR, TEXT_COLOR, GRID_COLOR, WATERMARK_COLOR  # noqa: E402

warnings.filterwarnings("ignore")
plt.style.use('dark_background')

UP_COLOR = '#2ee66b'
DOWN_COLOR = '#ff5c6c'
NEUTRAL_COLOR = '#40c0ff'

# clave -> (etiqueta, tiene signo, columna calculada)
METRICS = {
    "volumen": ("Volumen (acciones)", False, "Volumen"),
    "volumen_nominal": ("Volumen nominal (precio × volumen)", False, "VolNominal"),
    "volumen_relativo": ("Volumen relativo (vs promedio 20 ruedas)", False, "VolRel"),
    "rango": ("Rango máximo–mínimo (%)", False, "Rango"),
    "var_cierre_ant": ("Variación vs cierre anterior (%)", True, "VarCierre"),
    "apertura_cierre": ("Distancia apertura → cierre (%)", True, "AperCierre"),
    "gap": ("Gap de apertura vs cierre anterior (%)", True, "Gap"),
}

# frase para el título según métrica y modo (descriptiva, sin conclusiones)
_TITLE_PHRASE = {
    "volumen": "mayor volumen",
    "volumen_nominal": "mayor volumen nominal",
    "volumen_relativo": "mayor volumen relativo",
    "rango": "mayor rango máximo–mínimo",
    ("var_cierre_ant", "abs"): "mayor variación vs cierre anterior (absoluta)",
    ("var_cierre_ant", "up"): "mayores subas vs cierre anterior",
    ("var_cierre_ant", "down"): "mayores bajas vs cierre anterior",
    ("apertura_cierre", "abs"): "mayor distancia apertura → cierre (absoluta)",
    ("apertura_cierre", "up"): "mayores subas apertura → cierre",
    ("apertura_cierre", "down"): "mayores bajas apertura → cierre",
    ("gap", "abs"): "mayor gap de apertura (absoluto)",
    ("gap", "up"): "mayores gaps alcistas de apertura",
    ("gap", "down"): "mayores gaps bajistas de apertura",
}

_DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
PERIODS = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "MAX"]


# ===================== DATOS =====================
def _period_start(period, end):
    period = period.upper()
    if period == "MAX":
        return pd.Timestamp("1900-01-01")
    if period == "YTD":
        return pd.Timestamp(year=end.year - 1, month=12, day=31)
    days = {"1M": 31, "3M": 92, "6M": 183, "1Y": 365, "2Y": 730, "5Y": 1826, "10Y": 3652}.get(period, 365)
    return end - pd.Timedelta(days=days)


def fetch_ohlcv(ticker, start, end):
    """OHLCV diario (ajustado por splits/dividendos). Devuelve DataFrame o vacío."""
    df = yf_download(ticker, start=start.strftime('%Y-%m-%d'),
                     end=(end + pd.Timedelta(days=2)).strftime('%Y-%m-%d'),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    need = ["Open", "High", "Low", "Close"]
    if any(c not in df.columns for c in need):
        return pd.DataFrame()
    if "Volume" not in df.columns:
        df["Volume"] = np.nan
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[need + ["Volume"]].dropna(subset=["Close"])


def _drop_partial_session(df):
    """Si la última rueda es la de hoy en NY y todavía no cerró, se descarta (volumen/rango incompletos)."""
    now_ny = pd.Timestamp.now(tz="America/New_York")
    if df.empty or now_ny.weekday() >= 5:
        return df
    if df.index[-1].normalize() == pd.Timestamp(now_ny.date()) and now_ny.hour < 16:
        return df.iloc[:-1]
    return df


def compute_daily_metrics(df):
    out = pd.DataFrame(index=df.index)
    prev_close = df["Close"].shift(1)
    vol = df["Volume"].where(df["Volume"] > 0)
    out["Volumen"] = vol
    out["VolNominal"] = df["Close"] * vol
    out["VolRel"] = vol / vol.shift(1).rolling(20, min_periods=10).mean()
    out["Rango"] = (df["High"] / df["Low"] - 1) * 100
    out["VarCierre"] = (df["Close"] / prev_close - 1) * 100
    out["AperCierre"] = (df["Close"] / df["Open"] - 1) * 100
    out["Gap"] = (df["Open"] / prev_close - 1) * 100
    out["Cierre"] = df["Close"]
    return out.replace([np.inf, -np.inf], np.nan)


def top_days_table(ticker, metric="volumen", n=10, mode="abs", period="1Y",
                   start_date=None, end_date=None, include_partial=False):
    """Devuelve (tabla top N ordenada, métricas diarias del rango, df OHLCV) o (None, None, None)."""
    if metric not in METRICS:
        raise ValueError(f"Métrica desconocida: {metric}")
    label, signed, col = METRICS[metric]

    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(datetime.now().date())
    start = pd.Timestamp(start_date) if start_date else _period_start(period, end)
    buffer_start = start - pd.Timedelta(days=45)  # para cierre anterior y promedio de 20 ruedas

    df = fetch_ohlcv(ticker, buffer_start, end)
    if df.empty:
        print(f"AVISO: sin datos para {ticker}")
        return None, None, None
    if not include_partial:
        df = _drop_partial_session(df)

    m = compute_daily_metrics(df)
    m = m[(m.index >= start) & (m.index <= end + pd.Timedelta(days=1))]
    series = m[col].dropna()
    if series.empty:
        print(f"AVISO: {ticker} no tiene datos para la métrica {label} en ese rango")
        return None, None, None

    if signed:
        if mode == "up":
            series = series.sort_values(ascending=False)
        elif mode == "down":
            series = series.sort_values(ascending=True)
        else:
            series = series.reindex(series.abs().sort_values(ascending=False).index)
    else:
        series = series.sort_values(ascending=False)

    top = m.loc[series.index[:max(1, int(n))]].copy()
    top.insert(0, "Valor", series.iloc[:len(top)])
    return top, m, df


# ===================== FORMATO =====================
def _fmt_big(x):
    ax = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if ax >= div:
            return f"{x / div:.2f}{suf}"
    return f"{x:,.0f}"


def _fmt_value(metric, v):
    if metric == "volumen":
        return _fmt_big(v)
    if metric == "volumen_nominal":
        return "$" + _fmt_big(v)
    if metric == "volumen_relativo":
        return f"×{v:.2f}"
    if metric == "rango":
        return f"{v:.2f}%"
    return f"{v:+.2f}%"


def _title_phrase(metric, mode):
    return _TITLE_PHRASE.get((metric, mode)) or _TITLE_PHRASE.get(metric) or METRICS[metric][0]


# ===================== GRÁFICO =====================
def plot_top_days(top, ticker, metric, mode, n_total, start, end, output_path="top_ruedas.png"):
    label, signed, col = METRICS[metric]
    k = len(top)

    fig_h = max(5.2, 0.62 * k + 2.6)
    fig = plt.figure(figsize=(13, fig_h), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    vals = top["Valor"].values
    y = np.arange(k)[::-1]  # el #1 arriba
    if signed:
        colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in vals]
    else:
        colors = [NEUTRAL_COLOR] * k
    ax.barh(y, vals, color=colors, alpha=0.9, height=0.68, zorder=3)

    span = np.nanmax(np.abs(vals)) if k else 1
    pad = span * 0.012
    for yi, (dt, row) in zip(y, top.iterrows()):
        v = row["Valor"]
        txt = _fmt_value(metric, v)
        ha = "left" if v >= 0 else "right"
        xpos = v + (pad if v >= 0 else -pad)
        ax.text(xpos, yi, txt, va="center", ha=ha, fontsize=14, fontweight="bold",
                color=TEXT_COLOR, zorder=5)
        # en métricas de volumen/rango, columna a la derecha con la variación del día
        if not signed and pd.notna(row.get("VarCierre", np.nan)):
            ax.text(0.995, yi, f"{row['VarCierre']:+.2f}% día", transform=ax.get_yaxis_transform(),
                    va="center", ha="right", fontsize=11, color="#9a9ac0", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{dt:%d/%m/%Y} ({_DIAS[dt.weekday()]})" for dt in top.index],
                       fontsize=12, color=TEXT_COLOR)
    lo, hi = np.nanmin(vals), np.nanmax(vals)
    left = min(0, lo) - (abs(lo) * 0.18 if lo < 0 else 0)
    right = max(0, hi) + abs(hi) * (0.45 if not signed else 0.2)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.7, k - 0.3)
    if signed:
        ax.axvline(0, color='#ff4d4d', linewidth=1.2, linestyle='--', alpha=0.8, zorder=4)
    ax.grid(True, axis="x", color=GRID_COLOR, linestyle=':', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT_COLOR)
        ax.spines[sp].set_alpha(0.4)
    ax.tick_params(axis="x", colors="#c0c0ff", labelsize=10)
    ax.tick_params(axis="y", length=0)

    if metric in ("volumen", "volumen_nominal"):
        ax.xaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: ("$" if metric == "volumen_nominal" else "") + _fmt_big(v)))
    elif metric == "volumen_relativo":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"×{v:g}"))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))

    title = f"Top {k} ruedas por {_title_phrase(metric, mode)} — {ticker.upper()}"
    sub = f"{start:%d/%m/%Y} – {end:%d/%m/%Y} · {n_total} ruedas analizadas"
    fig.suptitle(f"{title}\n{sub}", fontsize=20, color=TEXT_COLOR, fontweight="bold", y=0.985)
    fig.text(0.5, 0.5, "@MTaurus_ok", fontsize=90, color=WATERMARK_COLOR,
             ha="center", va="center", rotation=30, zorder=999)

    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=PAPER_COLOR)
    plt.close(fig)
    return output_path


# ===================== RUNNER =====================
def run_top_days(ticker, metric="volumen", n=10, mode="abs", period="1Y",
                 start_date=None, end_date=None, include_partial=False,
                 return_table=False, output_path="top_ruedas.png"):
    ticker = ticker.strip().upper()
    top, m, df = top_days_table(ticker, metric, n, mode, period, start_date, end_date, include_partial)
    if top is None:
        return (None, None) if return_table else None
    label, signed, col = METRICS[metric]
    n_total = int(m[col].notna().sum())
    start = m.index.min()
    end = m.index.max()
    path = plot_top_days(top, ticker, metric, mode if signed else "abs", n_total, start, end, output_path)

    if return_table:
        table = top[["Valor", "Cierre", "VarCierre", "Volumen", "Rango", "AperCierre", "Gap"]].copy()
        table.index = table.index.strftime("%d/%m/%Y")
        table.index.name = "Fecha"
        table.columns = [label, "Cierre", "Var. vs cierre ant. (%)", "Volumen", "Rango máx–mín (%)",
                         "Apertura→Cierre (%)", "Gap apertura (%)"]
        return path, table
    return path


# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--metric", default="volumen", choices=list(METRICS))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--mode", default="abs", choices=["abs", "up", "down"])
    parser.add_argument("--period", default="1Y", choices=PERIODS)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--include_partial", action="store_true")
    args = parser.parse_args()
    p = run_top_days(args.ticker, args.metric, args.n, args.mode, args.period,
                     args.start, args.end, args.include_partial)
    if p:
        print(os.path.abspath(p))
