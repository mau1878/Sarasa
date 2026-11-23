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
HIGHLIGHT_COLOR = '#b78cff'   # barras cuya fecha cae dentro del rango a destacar

# clave -> (etiqueta, tiene signo, columna calculada)
METRICS = {
    "volumen": ("Volumen (acciones)", False, "Volumen"),
    "volumen_nominal": ("Volumen nominal (precio × volumen)", False, "VolNominal"),
    "volumen_relativo": ("Volumen relativo (vs promedio de los 20 períodos previos)", False, "VolRel"),
    "rango": ("Rango máximo–mínimo (%)", False, "Rango"),
    "var_cierre_ant": ("Variación vs cierre anterior (%)", True, "VarCierre"),
    "apertura_cierre": ("Distancia apertura → cierre (%)", True, "AperCierre"),
    "gap": ("Gap de apertura vs cierre anterior (%)", True, "Gap"),
    # --- solo en modo ratio A/B (dos instrumentos, únicamente ruedas con datos de ambos) ---
    "ratio_var": ("Variación del ratio vs rueda anterior en común (%)", True, "RatioVar"),
    "ratio_desvio": ("Desvío del ratio vs su media de las 20 ruedas previas (%)", True, "RatioDesv"),
    "ratio_nivel": ("Nivel del ratio", False, "RatioNivel"),
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
    ("ratio_var", "abs"): "mayor variación del ratio vs rueda anterior en común (absoluta)",
    ("ratio_var", "up"): "mayores subas del ratio vs rueda anterior en común",
    ("ratio_var", "down"): "mayores bajas del ratio vs rueda anterior en común",
    ("ratio_desvio", "abs"): "mayor desvío del ratio vs su media de 20 períodos (absoluto)",
    ("ratio_desvio", "up"): "mayores desvíos al alza del ratio vs su media de 20 períodos",
    ("ratio_desvio", "down"): "mayores desvíos a la baja del ratio vs su media de 20 períodos",
    "ratio_nivel": "mayor nivel del ratio",
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


def _to_weekly(df, include_partial=False):
    """Agrupa OHLCV diario en semanas (lun–vie). El índice es la última rueda de cada semana y
    'Desde' guarda la primera. Si no se incluyen datos parciales, se descarta la semana en curso incompleta."""
    d = df.copy()
    d["_d"] = d.index
    g = d.groupby(d.index.to_period("W-FRI"))
    w = pd.DataFrame({
        "Open": g["Open"].first(), "High": g["High"].max(), "Low": g["Low"].min(),
        "Close": g["Close"].last(), "Volume": g["Volume"].sum(min_count=1),
        "Desde": g["_d"].first(), "_fin": g["_d"].last(),
    })
    w.index = pd.DatetimeIndex(w.pop("_fin"))
    if not include_partial and len(w):
        hoy = pd.Timestamp(pd.Timestamp.now(tz="America/New_York").date())
        viernes = w.index[-1].normalize() + pd.Timedelta(days=4 - w.index[-1].weekday())
        if w.index[-1].weekday() < 4 and hoy <= viernes:      # semana todavía en curso
            w = w.iloc[:-1]
    return w


def _to_weekly_pair(c, include_partial=False):
    """Igual que _to_weekly pero para el par alineado (CierreA, CierreB, Close=ratio): usa el último cierre
    COMÚN de cada semana (el ratio semanal sale de esos cierres)."""
    d = c.copy()
    d["_d"] = d.index
    g = d.groupby(d.index.to_period("W-FRI"))
    w = pd.DataFrame({"CierreA": g["CierreA"].last(), "CierreB": g["CierreB"].last(), "Close": g["Close"].last(),
                      "Desde": g["_d"].first(), "_fin": g["_d"].last()})
    w.index = pd.DatetimeIndex(w.pop("_fin"))
    if not include_partial and len(w):
        hoy = pd.Timestamp(pd.Timestamp.now(tz="America/New_York").date())
        viernes = w.index[-1].normalize() + pd.Timedelta(days=4 - w.index[-1].weekday())
        if w.index[-1].weekday() < 4 and hoy <= viernes:
            w = w.iloc[:-1]
    return w


def compute_ratio_metrics(c):
    """c: DataFrame con CierreA, CierreB y Close (= factor × A / B), solo períodos con datos de ambos."""
    out = pd.DataFrame(index=c.index)
    r = c["Close"]
    prev = r.shift(1)                                   # cierre anterior EN COMÚN
    for col in ("Volumen", "VolNominal", "VolRel", "Rango", "AperCierre", "Gap"):
        out[col] = np.nan
    out["RatioNivel"] = r
    out["RatioVar"] = (r / prev - 1) * 100
    out["RatioDesv"] = (r / prev.rolling(20, min_periods=10).mean() - 1) * 100
    out["VarCierre"] = out["RatioVar"]                  # sentido del día (colores / columna contextual)
    out["Cierre"] = r
    out["CierreA"] = c["CierreA"]
    out["CierreB"] = c["CierreB"]
    if "Desde" in c.columns:
        out["Desde"] = c["Desde"]
    return out.replace([np.inf, -np.inf], np.nan)


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
    if "Desde" in df.columns:
        out["Desde"] = df["Desde"]
    return out.replace([np.inf, -np.inf], np.nan)


def top_days_table(ticker, metric="volumen", n=10, mode="abs", period="1Y",
                   start_date=None, end_date=None, include_partial=False, frequency="D",
                   ticker2=None, factor=1.0):
    """Devuelve (tabla top N ordenada, métricas diarias del rango, df OHLCV) o (None, None, None)."""
    if metric not in METRICS:
        raise ValueError(f"Métrica desconocida: {metric}")
    label, signed, col = METRICS[metric]

    end = pd.Timestamp(end_date) if end_date else pd.Timestamp(datetime.now().date())
    start = pd.Timestamp(start_date) if start_date else _period_start(period, end)
    weekly = str(frequency).upper().startswith("W")
    buffer_start = start - pd.Timedelta(days=170 if weekly else 45)  # cierre anterior y promedio de 20 períodos

    if ticker2:
        if not metric.startswith("ratio_"):
            raise ValueError("En modo ratio hay que elegir una métrica ratio_*")
        dfa, dfb = fetch_ohlcv(ticker, buffer_start, end), fetch_ohlcv(ticker2, buffer_start, end)
        for tk, d_ in ((ticker, dfa), (ticker2, dfb)):
            if d_.empty:
                print(f"AVISO: sin datos para {tk}")
                return None, None, None
        if not include_partial:
            dfa, dfb = _drop_partial_session(dfa), _drop_partial_session(dfb)
        for d_ in (dfa, dfb):
            d_.index = d_.index.normalize()
        # Solo ruedas con datos en AMBOS instrumentos
        df = pd.DataFrame({"CierreA": dfa["Close"], "CierreB": dfb["Close"]}).dropna()
        if df.empty:
            print(f"AVISO: {ticker} y {ticker2} no tienen ruedas en común en ese rango")
            return None, None, None
        df["Close"] = float(factor) * df["CierreA"] / df["CierreB"]
        if weekly:
            df = _to_weekly_pair(df, include_partial)
        m = compute_ratio_metrics(df)
    else:
        if metric.startswith("ratio_"):
            raise ValueError("Las métricas ratio_* requieren un segundo ticker")
        df = fetch_ohlcv(ticker, buffer_start, end)
        if df.empty:
            print(f"AVISO: sin datos para {ticker}")
            return None, None, None
        if not include_partial:
            df = _drop_partial_session(df)
        if weekly:
            df = _to_weekly(df, include_partial)
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
    """Número grande abreviado con ~3 cifras significativas: 819M, 62.6M, 5.49M."""
    ax_ = abs(x)
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if ax_ >= div:
            v = x / div
            a = abs(v)
            return (f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}") + suf
    return f"{x:,.0f}"


def _fmt_axis(x):
    """Etiqueta de eje sin decimales sobrantes: 200M, 1B."""
    if x == 0:
        return "0"
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:g}{suf}"
    return f"{x:g}"


def _fmt_value(metric, v):
    if metric == "volumen":
        return _fmt_big(v)
    if metric == "volumen_nominal":
        return "$" + _fmt_big(v)
    if metric == "volumen_relativo":
        return f"×{v:.2f}"
    if metric == "rango":
        return f"{v:.2f}%"
    if metric == "ratio_nivel":
        a = abs(v)
        return f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.3f}"   # sin separador de miles
    return f"{v:+.2f}%"


def _title_phrase(metric, mode):
    return _TITLE_PHRASE.get((metric, mode)) or _TITLE_PHRASE.get(metric) or METRICS[metric][0]


# ===================== GRÁFICO =====================
GOLD = '#ffd166'
MUTED = '#9a9ac0'


def plot_top_days(top, ticker, metric, mode, n_total, start, end, output_path="top_ruedas.png",
                  ref_value=None, sort_by_date=False, highlight_start=None, highlight_end=None,
                  fwd_ret=None, fwd_label=None, recency_gradient=False, frequency="D", ratio_mode=False):
    weekly = str(frequency).upper().startswith("W")
    unit, unit1 = ("semanas", "semana") if weekly else ("ruedas", "rueda")
    label, signed, col = METRICS[metric]
    top = top.copy()
    top["Rank"] = np.arange(1, len(top) + 1)          # puesto según la métrica
    if sort_by_date:
        top = top.sort_index()                         # cronológico (el puesto se mantiene en la etiqueta)
    k = len(top)

    fig_h = max(5.4, 0.66 * k + 2.8)
    fig = plt.figure(figsize=(13.5, fig_h), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    vals = top["Valor"].values
    y = np.arange(k)[::-1]  # la primera fila arriba

    # Color de las barras: con signo -> según el valor; volumen/rango -> según el sentido de la rueda
    if signed:
        colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in vals]
    else:
        vc = top["VarCierre"].values
        colors = [NEUTRAL_COLOR if pd.isna(v) else (UP_COLOR if v >= 0 else DOWN_COLOR) for v in vc]
    # Destacar con otro color las ruedas cuya fecha cae dentro de [highlight_start, highlight_end]
    n_hl, hl_label = 0, None
    if highlight_start is not None or highlight_end is not None:
        hs = pd.Timestamp(highlight_start) if highlight_start is not None else top.index.min()
        he = pd.Timestamp(highlight_end) if highlight_end is not None else top.index.max()
        if hs > he:
            hs, he = he, hs
        in_rng = [(hs <= pd.Timestamp(d).normalize() <= he) for d in top.index]
        colors = [HIGHLIGHT_COLOR if f else c for f, c in zip(in_rng, colors)]
        n_hl = int(sum(in_rng))
        hl_label = f"{hs:%d/%m/%Y} – {he:%d/%m/%Y}: {n_hl} de {k}"
    # Degradé por antigüedad: la rueda más reciente del top con el color pleno y la más antigua más desvanecida
    if recency_gradient and k > 1:
        from matplotlib.colors import to_rgb
        ts = np.array([pd.Timestamp(d).value for d in top.index], dtype=float)
        frac = (ts - ts.min()) / ((ts.max() - ts.min()) or 1.0)       # 0 = más antigua, 1 = más reciente
        bg = np.array(to_rgb(BG_COLOR))
        colors = [tuple(np.array(to_rgb(c)) * (0.58 + 0.42 * f) + bg * (0.42 - 0.42 * f))
                  for c, f in zip(colors, frac)]
    ax.barh(y, vals, color=colors, alpha=0.88, height=0.66, zorder=3)

    lo, hi = np.nanmin(vals), np.nanmax(vals)
    span = max(abs(lo), abs(hi)) or 1
    pad = span * 0.012
    for yi, (dt, row) in zip(y, top.iterrows()):
        v = row["Valor"]
        ax.text(v + (pad if v >= 0 else -pad), yi, _fmt_value(metric, v),
                va="center", ha="left" if v >= 0 else "right",
                fontsize=15, fontweight="bold", color="#ffffff", zorder=5)

    # Columna de contexto fuera del área de barras, con encabezado y color según signo
    if signed and ratio_mode:
        head = "Ratio"
        ctx = [(_fmt_value("ratio_nivel", r["Cierre"]) if pd.notna(r["Cierre"]) else "–", MUTED) for _, r in top.iterrows()]
    elif signed:
        head = "Volumen"
        ctx = [(_fmt_big(r["Volumen"]) if pd.notna(r["Volumen"]) else "–", MUTED) for _, r in top.iterrows()]
    else:
        head = "Var. de la semana" if weekly else "Var. del día"
        ctx = []
        for _, r in top.iterrows():
            v = r["VarCierre"]
            ctx.append(("–", MUTED) if pd.isna(v) else (f"{v:+.2f}%", UP_COLOR if v >= 0 else DOWN_COLOR))
    ax.text(1.015, 1.0 + 0.012, head, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color=MUTED, fontweight="bold")
    for yi, (txt, colr) in zip(y, ctx):
        ax.text(1.015, yi, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=14, fontweight="bold", color=colr, zorder=5)

    # Columna extra: retorno de las N ruedas siguientes (qué pasó después)
    if fwd_ret is not None:
        ax.text(1.115, 1.0 + 0.012, fwd_label or "Después", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=12, color=MUTED, fontweight="bold")
        for yi, (dt, _r) in zip(y, top.iterrows()):
            fv = fwd_ret.get(dt, np.nan)
            txt, colr = ("–", MUTED) if pd.isna(fv) else (f"{fv:+.2f}%", UP_COLOR if fv >= 0 else DOWN_COLOR)
            ax.text(1.115, yi, txt, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=14, fontweight="bold", color=colr, zorder=5)

    # Puesto + fecha
    ax.set_yticks(y)
    if weekly:
        ylabels = [f"#{int(r['Rank'])}  {pd.Timestamp(r['Desde']):%d/%m}–{dt:%d/%m/%Y}" for dt, r in top.iterrows()]
    else:
        ylabels = [f"#{int(r['Rank'])}  {dt:%d/%m/%Y} ({_DIAS[dt.weekday()]})" for dt, r in top.iterrows()]
    ax.set_yticklabels(ylabels, fontsize=13, color=TEXT_COLOR)

    left = min(0, lo) - (abs(lo) * 0.16 if lo < 0 else 0)
    right = max(0, hi) + abs(hi) * (0.12 if not signed else 0.30)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.65, k - 0.35)

    if signed:
        ax.axvline(0, color='#ff4d4d', linewidth=1.2, linestyle='--', alpha=0.8, zorder=4)
    if ref_value is not None and np.isfinite(ref_value) and not signed:
        ax.axvline(ref_value, color=GOLD, linewidth=1.3, linestyle=(0, (4, 3)), alpha=0.9, zorder=4)
        ax.text(ref_value, 1.012, f" Mediana del período: {_fmt_value(metric, ref_value)}",
                transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=11, color=GOLD)

    ax.grid(True, axis="x", color=GRID_COLOR, linestyle=':', linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(TEXT_COLOR)
        ax.spines[sp].set_alpha(0.35)
    ax.tick_params(axis="x", colors="#c0c0ff", labelsize=11)
    ax.tick_params(axis="y", length=0, pad=8)

    if metric in ("volumen", "volumen_nominal"):
        pre = "$" if metric == "volumen_nominal" else ""
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pre + _fmt_axis(v)))
    elif metric == "volumen_relativo":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"×{v:g}"))
    elif metric == "ratio_nivel":
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    else:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}%"))

    # Leyenda de colores (solo cuando el color indica el sentido de la rueda)
    from matplotlib.patches import Patch
    handles = []
    if not signed:
        handles += [Patch(color=UP_COLOR, alpha=0.88, label=f"{unit1.capitalize()} con suba"),
                    Patch(color=DOWN_COLOR, alpha=0.88, label=f"{unit1.capitalize()} con baja")]
    if hl_label:
        handles.append(Patch(color=HIGHLIGHT_COLOR, alpha=0.88, label=hl_label))
    if recency_gradient and k > 1:
        from matplotlib.lines import Line2D
        handles.append(Line2D([], [], color="none", label="Más reciente = color más intenso"))
    if handles:
        below = False
        if not signed and k >= 3:
            # Si las barras de abajo llegan cerca del borde derecho, la leyenda se pasa debajo del eje
            xl, xr = ax.get_xlim()
            below = bool((np.nanmax(vals[-3:]) - xl) / ((xr - xl) or 1) > 0.62)
        if below:
            leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=len(handles),
                            frameon=False, fontsize=11, labelcolor=TEXT_COLOR)
        else:
            leg = ax.legend(handles=handles, loc="lower right" if not signed else "best",
                            frameon=False, fontsize=11, labelcolor=TEXT_COLOR)
        leg.set_zorder(6)

    orden = "ordenadas por fecha" if sort_by_date else "ordenadas por valor"
    phrase = _title_phrase(metric, mode)
    if weekly:
        phrase = phrase.replace("cierre anterior", "cierre semanal anterior")
    title = f"Top {k} {unit} por {phrase} — {ticker.upper()}"
    sub = f"{start:%d/%m/%Y} – {end:%d/%m/%Y} · {n_total} {unit}{' en común' if ratio_mode else ''} analizadas · {orden}"
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
                 return_table=False, output_path="top_ruedas.png", sort_by_date=False,
                 highlight_start=None, highlight_end=None, fwd_sessions=0, recency_gradient=False,
                 frequency="D", ticker2=None, factor=1.0):
    ticker = ticker.strip().upper()
    ticker2 = ticker2.strip().upper() if ticker2 and ticker2.strip() else None
    ratio = bool(ticker2)
    factor = float(factor or 1.0)
    weekly = str(frequency).upper().startswith("W")
    top, m, df = top_days_table(ticker, metric, n, mode, period, start_date, end_date, include_partial, frequency,
                                ticker2=ticker2, factor=factor)
    if top is None:
        return (None, None) if return_table else None
    label, signed, col = METRICS[metric]
    n_total = int(m[col].notna().sum())
    start = m.index.min()
    end = m.index.max()
    ref = float(m[col].median()) if not signed else None
    fwd_ret = fwd_label = None
    k_fwd = int(fwd_sessions or 0)
    if k_fwd > 0:
        closes = df["Close"]
        fwd_ret = (closes.shift(-k_fwd) / closes - 1) * 100            # retorno del cierre de t al cierre de t+k
        if weekly:
            fwd_label = "Semana siguiente" if k_fwd == 1 else f"A {k_fwd} semanas"
        else:
            fwd_label = "Rueda siguiente" if k_fwd == 1 else f"A {k_fwd} ruedas"
        if ratio:
            fwd_label = "Ratio: " + fwd_label[0].lower() + fwd_label[1:]
    tk_label = f"{ticker}/{ticker2}" + (f" ×{factor:g}" if ratio and factor != 1 else "") if ratio else ticker
    path = plot_top_days(top, tk_label, metric, mode if signed else "abs", n_total, start, end, output_path,
                         ref_value=ref, sort_by_date=sort_by_date,
                         highlight_start=highlight_start, highlight_end=highlight_end,
                         fwd_ret=fwd_ret, fwd_label=fwd_label, recency_gradient=recency_gradient,
                         frequency=frequency, ratio_mode=ratio)

    if return_table:
        if ratio:
            table = top[["Valor", "Cierre", "VarCierre", "CierreA", "CierreB"]].copy()
        else:
            table = top[["Valor", "Cierre", "VarCierre", "Volumen", "Rango", "AperCierre", "Gap"]].copy()
        if fwd_ret is not None:
            table[fwd_label] = [fwd_ret.get(d, np.nan) for d in top.index]
        table.index = table.index.strftime("%d/%m/%Y")
        table.index.name = "Fecha (fin de semana)" if weekly else "Fecha"
        if ratio:
            cols = [label, "Ratio", "Var. del ratio (%)", f"Cierre {ticker}", f"Cierre {ticker2}"]
        else:
            cols = [label, "Cierre", "Var. vs cierre ant. (%)", "Volumen", "Rango máx–mín (%)",
                    "Apertura→Cierre (%)", "Gap apertura (%)"]
        table.columns = cols + ([f"{fwd_label} (%)"] if fwd_ret is not None else [])
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
    parser.add_argument("--sort_by_date", action="store_true", help="Ordenar las barras por fecha en vez de por valor")
    parser.add_argument("--highlight_start", default=None, help="Destacar con otro color las ruedas desde esta fecha (YYYY-MM-DD)")
    parser.add_argument("--highlight_end", default=None, help="...hasta esta fecha (YYYY-MM-DD)")
    parser.add_argument("--ticker2", default=None, help="Segundo instrumento: calcula el ratio TICKER/TICKER2 (métricas ratio_*)")
    parser.add_argument("--factor", type=float, default=1.0, help="Multiplicador del ratio (p. ej. ratio de conversión del ADR)")
    parser.add_argument("--frequency", default="D", choices=["D", "W"], help="D = ruedas diarias, W = semanas")
    parser.add_argument("--fwd_sessions", type=int, default=0, help="Mostrar el retorno de las N ruedas siguientes (0 = no)")
    parser.add_argument("--recency_gradient", action="store_true", help="Degradé: más reciente = color más intenso")
    args = parser.parse_args()
    p = run_top_days(args.ticker, args.metric, args.n, args.mode, args.period,
                     args.start, args.end, args.include_partial, sort_by_date=args.sort_by_date,
                     highlight_start=args.highlight_start, highlight_end=args.highlight_end,
                     fwd_sessions=args.fwd_sessions, recency_gradient=args.recency_gradient,
                     frequency=args.frequency, ticker2=args.ticker2, factor=args.factor)
    if p:
        print(os.path.abspath(p))
