import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mtaurus_style  # noqa: F401  (tipografía compartida)
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os
import json
from datetime import datetime, timedelta
import warnings
import textwrap

# Ignorar warnings de yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===================== CONFIG =====================
# Tema Taurus (misma paleta que heatmap_script / drawdown_plot / returns_evolution_plot)
from mtaurus_style import (  # noqa: E402  (paleta compartida)
    BG_COLOR, PAPER_COLOR, TEXT_COLOR, TICK_COLOR, GRID_COLOR, WATERMARK_COLOR, ZERO_LINE_COLOR, DIV_COLORS,
)
TICKER_FONT_BOOST = 2   # puntos extra del ticker respecto del valor (subir/bajar a gusto)
LABEL_GAP_PT = 5        # separación entre ticker y valor
NOTE_FONT_SIZE = 13     # tamaño del subtítulo/aclaración opcional
AUTO_HORIZONTAL_MAX = 35  # orientation='auto': horizontal hasta esta cantidad de barras, vertical por encima
AUTO_DUAL_PAIRED_MAX = 24  # 2 fechas de inicio: barras pareadas hasta esta cantidad de tickers; por encima, barra ancha + fina
LEGEND_FONT_SIZE = 14      # tamaño de la leyenda (2 fechas de inicio)
plt.style.use('dark_background')

# ==================== DIVIDENDS ADJUSTMENTS ====================
DIVIDENDS = {
    'BYMA.BA': [
        (datetime(2026, 4, 14), 22.68)
    ]
    # Podés agregar más excepciones acá abajo:
    # 'GGAL.BA': [(datetime(2025, 10, 5), 15.00)],
}

def apply_dividend_adjustments(df):
    if df is None or df.empty:
        return df
    for ticker in df.columns:
        if ticker in DIVIDENDS:
            for ex_date, amount in DIVIDENDS[ticker]:
                mask = df.index <= ex_date
                df.loc[mask, ticker] -= amount
    return df

def load_ticker_names(json_path):
    mapping = {}
    if not os.path.exists(json_path):
        return mapping
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def extract_names(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str):
                        mapping[k.upper()] = v
                    elif isinstance(v, (dict, list)):
                        extract_names(v)
            elif isinstance(obj, list):
                pass
        extract_names(data)
    except Exception as e:
        print(f"Warning: No se pudo leer nombres del JSON: {e}")
    return mapping

def fetch_data(tickers, start_date, end_date):
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = tickers if isinstance(tickers, list) else [tickers]
        return apply_dividend_adjustments(data)
    except Exception:
        try:
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
            if isinstance(data, pd.Series):
                data = data.to_frame()
                data.columns = tickers if isinstance(tickers, list) else [tickers]
            return apply_dividend_adjustments(data)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

def fetch_ccl_proxy(start_date, end_date):
    try:
        ccl_data = yf.download(['YPFD.BA', 'YPF'], start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        if ccl_data.empty or 'YPFD.BA' not in ccl_data.columns or 'YPF' not in ccl_data.columns:
            return None
        ccl_ratio = ccl_data['YPFD.BA'] / ccl_data['YPF']
        ccl_ratio = ccl_ratio.interpolate(method='time').ffill().bfill()
        return ccl_ratio
    except Exception as e:
        print(f"Warning: Error fetching CCL: {e}")
        return None

def calculate_performance(data, reference_prices=None):
    data = data.dropna(how='all')
    if data.empty:
        return pd.Series()
    data_filled = data.ffill().bfill()
    if reference_prices is not None and not reference_prices.empty:
        first_price = reference_prices.reindex(data_filled.columns)
        first_price = first_price.fillna(data_filled.iloc[0])
    else:
        first_price = data_filled.iloc[0]
    last_price = data_filled.iloc[-1]
    returns = ((last_price - first_price) / first_price) * 100
    return returns.sort_values(ascending=False)

def _prepare_note(subtitle, est_width):
    """Parte el subtítulo opcional en líneas según el ancho estimado de la figura (en pulgadas).
    Devuelve (líneas, alto en puntos, alto extra en pulgadas a reservar arriba del gráfico)."""
    max_chars = max(30, int(est_width * 72 * 0.9 / (NOTE_FONT_SIZE * 0.55)))
    note_lines = []
    if subtitle and str(subtitle).strip():
        for para in str(subtitle).strip().splitlines():
            note_lines += textwrap.wrap(para.strip(), width=max_chars) or [""]
    note_h_pt = len(note_lines) * NOTE_FONT_SIZE * 1.6 if note_lines else 0   # ~alto real de una línea de matplotlib
    note_extra_in = (note_h_pt + 6) / 72 if note_lines else 0   # alto extra que pide arriba del gráfico
    return note_lines, note_h_pt, note_extra_in


def _draw_header(ax, title, period_label, note_lines, note_h_pt, base_off_pt=0):
    """Título, periodo y aclaración opcional, anclados al eje (centrados sobre las barras,
    separación fija en puntos). De arriba hacia abajo: título > periodo > aclaración (más chica,
    cursiva y más tenue). base_off_pt reserva espacio debajo (p. ej. para la leyenda de 2 fechas)."""
    period_off = 18 + base_off_pt + (note_h_pt + 4 if note_lines else 0)
    title_off = period_off + 28
    ax.annotate(title, xy=(0.5, 1), xycoords='axes fraction', xytext=(0, title_off), textcoords='offset points',
                ha='center', va='bottom', fontsize=22, fontweight='bold', color=TEXT_COLOR,
                annotation_clip=False)
    ax.annotate(period_label, xy=(0.5, 1), xycoords='axes fraction', xytext=(0, period_off), textcoords='offset points',
                ha='center', va='bottom', fontsize=15, color=TICK_COLOR, annotation_clip=False)
    if note_lines:
        ax.annotate("\n".join(note_lines), xy=(0.5, 1), xycoords='axes fraction', xytext=(0, 12 + base_off_pt),
                    textcoords='offset points', ha='center', va='bottom', fontsize=NOTE_FONT_SIZE,
                    fontstyle='italic', linespacing=1.0, color=TICK_COLOR, alpha=0.85,
                    annotation_clip=False)


def _fit_value_limits(fig, ax, items, vmin, vmax, horizontal):
    """Amplía los límites del eje de valores hasta que entren todas las etiquetas.
    items: lista de (anotación, positive). Las etiquetas están en puntos, no en unidades de datos,
    así que se mide el espacio real que ocupan y se itera hasta converger."""
    set_value_lim = ax.set_xlim if horizontal else ax.set_ylim
    pad = (vmax - vmin) * 0.03
    set_value_lim(vmin - pad, vmax + pad)
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        hi, lo = vmax, vmin
        for ann, positive in items:
            bb = ann.get_window_extent(renderer)
            p0 = inv.transform((bb.x0, bb.y0))
            p1 = inv.transform((bb.x1, bb.y1))
            c0, c1 = (p0[0], p1[0]) if horizontal else (p0[1], p1[1])
            if positive:
                hi = max(hi, c0, c1)
            else:
                lo = min(lo, c0, c1)
        span = hi - lo
        set_value_lim(lo - span * 0.02, hi + span * 0.02)


def create_bar_plot(performance, title, period_label, name_mapping,
                    filename="performance_chart.png", orientation="auto", subtitle=None):
    """Gráfico de barras de performance.

    orientation:
      'vertical'   barras verticales, etiquetas rotadas 90° (apaisado, mejor con muchas barras)
      'horizontal' barras horizontales, texto horizontal (más alto que ancho, mejor con pocas/medianas)
      'auto'       horizontal hasta AUTO_HORIZONTAL_MAX barras, vertical por encima

    subtitle: aclaración opcional (texto libre) que se muestra bajo la línea del periodo.
      Se ajusta sola al ancho del gráfico; con "\n" se fuerzan saltos de línea.
    """
    if performance.empty:
        return None
    df = performance.to_frame(name='Return')
    # Etiqueta en una sola línea: "TICKER  valor%" (sin saltos de línea)
    df['Display Name'] = [
        " ".join(str(name_mapping.get(t, t)).split()) for t in df.index
    ]

    # Mismo divergente apagado del heatmap: rojo - gris azulado - verde
    cmap = LinearSegmentedColormap.from_list("custom_div_dark", DIV_COLORS, N=256)
    vmin = df['Return'].min()
    vmax = df['Return'].max()
    if vmin >= 0: vmin = -0.01
    if vmax <= 0: vmax = 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    palette_colors = [cmap(norm(v)) for v in df['Return']]

    num_bars = len(df)
    if orientation == "auto":
        orientation = "horizontal" if num_bars <= AUTO_HORIZONTAL_MAX else "vertical"
    horizontal = orientation == "horizontal"

    # ---------- Subtítulo opcional: se parte en líneas según el ancho estimado de la figura ----------
    est_width = 12 if horizontal else max(12, min(22, 10 + num_bars * 0.5))
    note_lines, note_h_pt, note_extra_in = _prepare_note(subtitle, est_width)

    # ---------- Geometría y tamaños de fuente ----------
    if horizontal:
        fig_width = 12
        fig_height = min(20, max(7, 2.3 + 0.30 * num_bars))
        top_in, bottom_in, left_in, right_in = 1.35 + note_extra_in, 0.95, 0.35, 0.30
        axes_h_pt = (fig_height - top_in - bottom_in) * 72
        pitch_pt = axes_h_pt / num_bars                      # alto disponible por fila
        ticker_fs = min(18, max(11, 0.62 * pitch_pt))
        label_fs = ticker_fs - TICKER_FONT_BOOST
    else:
        calculated_width = 10 + (num_bars * 0.5)
        fig_width = max(12, min(22, calculated_width))
        fig_height = 10
        if num_bars > 50:
            label_fs = 12
        elif num_bars > 40:
            label_fs = 12.5
        elif num_bars > 25:
            label_fs = 13
        else:
            label_fs = 14
        ticker_fs = label_fs + TICKER_FONT_BOOST

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)
    positions = list(range(num_bars))
    values = list(df['Return'])

    # Franjas alternas muy tenues: separan una barra de la siguiente sin líneas
    band = ax.axhspan if horizontal else ax.axvspan
    for i in range(0, num_bars, 2):
        band(i - 0.5, i + 0.5, color=TICK_COLOR, alpha=0.04, linewidth=0, zorder=0)

    if horizontal:
        ax.barh(positions, values, color=palette_colors, height=0.78, zorder=3)
    else:
        ax.bar(positions, values, color=palette_colors, width=0.8, zorder=3)

    # ---------- Etiquetas pegadas al extremo de cada barra ----------
    # Vertical: arriba si >=0, abajo si <0, rotadas. Horizontal: a la derecha si >=0, a la izquierda si <0.
    # El ticker va más grande que el valor; se lee "TICKER  valor%".
    pairs = []  # (ticker_ann, value_ann, positive)
    for i, name, value in zip(positions, df['Display Name'], values):
        positive = value >= 0
        sign = 1 if positive else -1
        if horizontal:
            common = dict(xy=(value, i), ha='left' if positive else 'right', va='center', rotation=0)
            start = (4 * sign, 0)
        else:
            common = dict(xy=(i, value), ha='center', va='bottom' if positive else 'top', rotation=90)
            start = (0, 4 * sign)
        common.update(textcoords='offset points', color=TEXT_COLOR, fontweight='bold',
                      zorder=5, annotation_clip=False)
        t_ann = ax.annotate(name, xytext=start, fontsize=ticker_fs, **common)
        v_ann = ax.annotate(f'{value:.1f}%', xytext=start, fontsize=label_fs, **common)
        pairs.append((t_ann, v_ann, positive))

    # ---------- Línea del cero, grilla y ejes ----------
    zero_line = ax.axvline if horizontal else ax.axhline
    zero_line(0, color=ZERO_LINE_COLOR, linewidth=1.5, alpha=0.8, linestyle='--', zorder=4)
    ax.set_axisbelow(True)
    ax.grid(True, axis='x' if horizontal else 'y', color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)

    pct = plt.FuncFormatter(lambda x, _: f'{x:.0f}%')
    if horizontal:
        ax.set_ylim(num_bars - 0.3, -0.7)   # invertido: la mayor performance arriba
        ax.set_yticks([])                   # el ticker ya está en la etiqueta de cada barra
        ax.set_xlabel("Retorno (%)", fontsize=14, color=TEXT_COLOR, labelpad=10)
        ax.xaxis.set_major_formatter(pct)
        ax.tick_params(axis='x', colors=TICK_COLOR, labelsize=14)
        value_spine, hidden_spines = 'bottom', ['top', 'right', 'left']
    else:
        ax.set_xlim(-0.7, num_bars - 0.3)
        ax.set_xticks([])                   # el ticker ya está en la etiqueta de cada barra
        ax.set_ylabel("Retorno (%)", fontsize=14, color=TEXT_COLOR, labelpad=12)
        ax.yaxis.set_major_formatter(pct)
        ax.tick_params(axis='y', colors=TICK_COLOR, labelsize=14)
        value_spine, hidden_spines = 'left', ['top', 'right', 'bottom']

    # Solo queda el spine del eje de valores (el otro no marca nada: el cero lo marca la línea roja)
    ax.spines[hidden_spines].set_visible(False)
    ax.spines[value_spine].set_color(TEXT_COLOR)
    ax.spines[value_spine].set_linewidth(0.8)

    _draw_header(ax, title, period_label, note_lines, note_h_pt)

    ax.text(0.5, 0.5, '@MTaurus_ok', transform=ax.transAxes, fontsize=70 if horizontal else 90,
            color=WATERMARK_COLOR, ha='center', va='center', rotation=30, zorder=999)

    # Márgenes fijos (no tight_layout): así el tamaño del eje no depende de las etiquetas
    # y el ajuste del eje de valores de más abajo converge siempre igual.
    if horizontal:
        fig.subplots_adjust(left=left_in / fig_width, right=1 - right_in / fig_width,
                            top=1 - top_in / fig_height, bottom=bottom_in / fig_height)
    else:
        fig.subplots_adjust(left=0.06, right=0.995, top=1 - (1.1 + note_extra_in) / fig_height, bottom=0.03)

    # ---------- Ajuste del eje de valores a las etiquetas ----------
    # Las etiquetas están en puntos, no en unidades de datos: se mide el espacio real que ocupan.
    # 1) Apilar ticker y valor según el largo real de cada texto (en puntos, a lo largo de la barra)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72.0 / fig.dpi

    def _along(ann):
        bb = ann.get_window_extent(renderer)
        return (bb.width if horizontal else bb.height) * px_to_pt

    def _offset(amount):
        return (amount, 0) if horizontal else (0, amount)

    for t_ann, v_ann, positive in pairs:
        if positive:   # ticker junto a la barra, valor más afuera
            v_ann.xyann = _offset(4 + _along(t_ann) + LABEL_GAP_PT)
        else:          # valor junto a la barra, ticker más afuera
            t_ann.xyann = _offset(-(4 + _along(v_ann) + LABEL_GAP_PT))

    # 2) Ampliar los límites del eje de valores hasta que entren todas las etiquetas
    _fit_value_limits(fig, ax, [(a, pos) for t, v, pos in pairs for a in (t, v)], vmin, vmax, horizontal)

    output_path = os.path.abspath(filename)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close()
    return output_path


def create_dual_bar_plot(perf_df, title, period_label, name_mapping, series_labels,
                         filename="performance_chart.png", subtitle=None, style="auto", value_tags=None):
    """Barras horizontales con DOS fechas de inicio (columnas 'A' y 'B' de perf_df, ya ordenadas).

    style:
      'paired'  dos barras por ticker: A sólida, B "fantasma" (relleno translúcido + borde).
                Mejor con pocos tickers.
      'bullet'  una fila por ticker: barra ancha para A y barra fina clara encima para B.
                Escala mejor a muchos tickers.
      'auto'    paired hasta AUTO_DUAL_PAIRED_MAX tickers, bullet por encima.

    value_tags: ('YTD','MTD') opcional; en 'paired' agrega esa abreviatura, más chica y tenue, después de cada valor.

    Siempre horizontal (con dos series el texto rotado 90° no entra). Las dos series comparten la
    misma escala de color, así que los colores son comparables entre A y B.
    """
    df = perf_df.dropna(how='all')
    if df.empty:
        return None
    n = len(df)
    if style == "auto":
        style = "paired" if n <= AUTO_DUAL_PAIRED_MAX else "bullet"
    paired = style == "paired"
    names = [" ".join(str(name_mapping.get(t, t)).split()) for t in df.index]
    vals_a, vals_b = df['A'].tolist(), df['B'].tolist()

    # Misma escala de color para las dos series
    allv = pd.concat([df['A'], df['B']]).dropna()
    vmin, vmax = float(allv.min()), float(allv.max())
    if vmin >= 0: vmin = -0.01
    if vmax <= 0: vmax = 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = LinearSegmentedColormap.from_list("custom_div_dark", DIV_COLORS, N=256)

    # ---------- Geometría ----------
    la, lb = series_labels
    legend_rows = 1 if len(la) + len(lb) <= 80 else 2
    legend_h_pt = LEGEND_FONT_SIZE * 1.9 * legend_rows + 12
    fig_width = 12
    note_lines, note_h_pt, note_extra_in = _prepare_note(subtitle, fig_width)
    top_in = 1.35 + note_extra_in + legend_h_pt / 72
    bottom_in, left_in, right_in = 0.95, 0.35, 0.30
    row_in = 0.52 if paired else 0.30                       # alto por ticker
    fig_height = min(26, max(7, top_in + bottom_in + row_in * n))
    pitch_pt = (fig_height - top_in - bottom_in) * 72 / n   # alto disponible por ticker
    if paired:
        ticker_fs = min(15, max(9.5, 0.80 * pitch_pt / 2))  # cada barra tiene media fila
    else:
        ticker_fs = min(18, max(11, 0.62 * pitch_pt))
    value_fs = ticker_fs - TICKER_FONT_BOOST

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Franjas alternas muy tenues: agrupan visualmente las barras de un mismo ticker
    for i in range(0, n, 2):
        ax.axhspan(i - 0.5, i + 0.5, color=TICK_COLOR, alpha=0.04, linewidth=0, zorder=0)

    # ---------- Barras ----------
    for i, (a, b) in enumerate(zip(vals_a, vals_b)):
        a_ok, b_ok = not pd.isna(a), not pd.isna(b)
        if paired:
            if a_ok:   # A: sólida
                ax.barh(i - 0.2, a, height=0.36, color=cmap(norm(a)), zorder=3)
            if b_ok:   # B: fantasma (relleno translúcido + borde más claro)
                c = cmap(norm(b))
                ax.barh(i + 0.2, b, height=0.36, facecolor=(c[0], c[1], c[2], 0.50),
                        edgecolor=_lighten(c, 0.30), linewidth=1.5, zorder=3)
        else:
            if a_ok:   # A: barra ancha
                ax.barh(i, a, height=0.78, color=cmap(norm(a)), zorder=3)
            if b_ok:   # B: barra fina clara encima
                ax.barh(i, b, height=0.30, color=TEXT_COLOR, alpha=0.92,
                        edgecolor=BG_COLOR, linewidth=0.6, zorder=4)

    # ---------- Etiquetas ----------
    # Cada grupo es un renglón de texto pegado al extremo de una barra; "parts" va en orden de lectura
    # (izquierda a derecha). A la derecha de la barra se apilan hacia afuera en ese orden; a la
    # izquierda, en orden inverso, para que el texto se lea igual en ambos lados.
    groups = []
    tag_fs = max(8, value_fs - 3)
    tag_color = (*mcolors.to_rgb(TICK_COLOR), 0.7)

    def tag_part(k):   # abreviatura del período tras el valor (solo con 'paired' y value_tags)
        return [(str(value_tags[k]), tag_color, tag_fs, 'normal')] if (paired and value_tags) else []

    for i, (nm, a, b) in enumerate(zip(names, vals_a, vals_b)):
        a_ok, b_ok = not pd.isna(a), not pd.isna(b)
        ticker_part = (nm, TEXT_COLOR, ticker_fs)
        if paired:
            if a_ok:
                groups.append(dict(y=i - 0.2, positive=a >= 0, anchor=a,
                                   parts=[ticker_part, (f'{a:.1f}%', TEXT_COLOR, value_fs)] + tag_part(0)))
            if b_ok:
                # el ticker también acompaña a B (un poco más tenue), así cada barra se identifica sola
                parts = [(nm, TICK_COLOR, ticker_fs), (f'{b:.1f}%', TEXT_COLOR, value_fs)] + tag_part(1)
                groups.append(dict(y=i + 0.2, positive=b >= 0, anchor=b, parts=parts))
        else:
            ticker_side = (a >= 0) if a_ok else (b >= 0)
            for side in (True, False):
                members = []
                if a_ok and (a >= 0) == side:
                    members.append((a, (f'{a:.1f}%', _lighten(cmap(norm(a)), 0.45), value_fs)))   # valor de A: tono de su barra, aclarado
                if b_ok and (b >= 0) == side:
                    members.append((b, (f'{b:.1f}%', '#ffffff', value_fs)))      # valor de B, blanco
                if not members:
                    continue
                parts = [p for _, p in members]
                if ticker_side == side:
                    parts.insert(0, ticker_part)
                anchor = max(v for v, _ in members) if side else min(v for v, _ in members)
                groups.append(dict(y=i, positive=side, anchor=anchor, parts=parts))

    all_anns = []   # (anotación, positive) para ajustar el eje
    for g in groups:
        g['anns'] = []
        for part in g['parts']:
            text, color, fs = part[:3]
            weight = part[3] if len(part) > 3 else 'bold'
            ann = ax.annotate(text, xy=(g['anchor'], g['y']), xytext=(0, 0), textcoords='offset points',
                              ha='left' if g['positive'] else 'right', va='center', color=color,
                              fontsize=fs, fontweight=weight, zorder=5, annotation_clip=False)
            g['anns'].append(ann)
            all_anns.append((ann, g['positive']))

    # ---------- Línea del cero, grilla y ejes ----------
    ax.axvline(0, color=ZERO_LINE_COLOR, linewidth=1.5, alpha=0.8, linestyle='--', zorder=4)
    ax.set_axisbelow(True)
    ax.grid(True, axis='x', color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)
    ax.set_ylim(n - 0.3, -0.7)   # invertido: la mayor performance arriba
    ax.set_yticks([])            # el ticker ya está en la etiqueta de cada barra
    ax.set_xlabel("Retorno (%)", fontsize=14, color=TEXT_COLOR, labelpad=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(axis='x', colors=TICK_COLOR, labelsize=14)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR)
    ax.spines['bottom'].set_linewidth(0.8)

    # ---------- Encabezado y leyenda ----------
    _draw_header(ax, title, period_label, note_lines, note_h_pt, base_off_pt=legend_h_pt)
    if paired:
        handles = [Patch(facecolor=TICK_COLOR, alpha=0.8, edgecolor='none'),
                   Patch(facecolor=(*mcolors.to_rgb(TICK_COLOR), 0.40), edgecolor=TICK_COLOR, linewidth=1.5)]
    else:
        handles = [Line2D([0], [0], color=TICK_COLOR, lw=11, alpha=0.55, solid_capstyle='butt'),
                   Line2D([0], [0], color='#ffffff', lw=4, solid_capstyle='butt')]
    ax.legend(handles, [la, lb], loc='lower center', bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3,
              ncol=2 if legend_rows == 1 else 1, frameon=False, fontsize=LEGEND_FONT_SIZE,
              labelcolor=TEXT_COLOR, handlelength=1.8, handleheight=0.9, columnspacing=2.4)

    ax.text(0.5, 0.5, '@MTaurus_ok', transform=ax.transAxes, fontsize=70,
            color=WATERMARK_COLOR, ha='center', va='center', rotation=30, zorder=999)

    # Márgenes fijos (no tight_layout), igual que en el gráfico de una sola fecha
    fig.subplots_adjust(left=left_in / fig_width, right=1 - right_in / fig_width,
                        top=1 - top_in / fig_height, bottom=bottom_in / fig_height)

    # ---------- Apilar los textos de cada barra según su ancho real ----------
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72.0 / fig.dpi
    for g in groups:
        widths = [a.get_window_extent(renderer).width * px_to_pt for a in g['anns']]
        if g['positive']:
            off = 4
            for ann, w in zip(g['anns'], widths):
                ann.xyann = (off, 0)
                off += w + LABEL_GAP_PT
        else:
            off = -4
            for ann, w in zip(reversed(g['anns']), reversed(widths)):
                ann.xyann = (off, 0)
                off -= w + LABEL_GAP_PT

    _fit_value_limits(fig, ax, all_anns, vmin, vmax, True)

    output_path = os.path.abspath(filename)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close()
    return output_path


def _lighten(color, amount=0.3):
    """Mezcla un color con blanco (amount=0 no cambia nada, 1 da blanco)."""
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


QUICK_PERIODS = ("YTD", "MTD", "WTD", "1Y", "2Y", "5Y")


def _period_spec(period, now):
    """(target_ref_date, start_date) de un período predefinido. target_ref_date es el último día cuyo
    cierre sirve de base (None = base en el primer dato desde start_date). Misma lógica de siempre."""
    target_ref_date = None
    if period == "YTD":
        target_ref_date = datetime(now.year - 1, 12, 31).date()
        start_date = (now.replace(month=1, day=1) - timedelta(days=15)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "1Y":
        start_date = now - timedelta(days=365)
    elif period == "2Y":
        start_date = now - timedelta(days=365 * 2)
    elif period == "5Y":
        start_date = now - timedelta(days=365 * 5)
    elif period == "MTD":
        if now.month == 1:
            prev_month_end = datetime(now.year - 1, 12, 31)
        else:
            prev_month_end = datetime(now.year, now.month, 1) - timedelta(days=1)
        target_ref_date = prev_month_end.date()
        start_date = (now - timedelta(days=70)).replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "WTD":
        weekday = now.weekday()
        days_back = (weekday - 4) % 7 + 7
        if weekday == 4:
            days_back = 7
        target_ref_date = (now - timedelta(days=days_back)).date()
        start_date = (now - timedelta(days=25)).replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        start_date = now - timedelta(days=365)
    return target_ref_date, start_date


# ===================== UPDATED RUNNER WITH CUSTOM START DATE =====================
def run_performance(
    tickers,
    period="YTD",
    group_name="Activos",
    tickers_file=None,
    ccl_adjust=False,
    custom_start_date=None,  # ← NEW
    orientation="auto",      # 'auto' | 'horizontal' | 'vertical'
    subtitle=None,           # aclaración opcional bajo el periodo (texto libre)
    custom_start_date_2=None,  # 2.ª fecha de inicio opcional: compara ambas en el mismo gráfico
    label_a=None,            # nombre opcional de la 1.ª fecha en la leyenda (solo con 2 fechas)
    label_b=None,            # nombre opcional de la 2.ª fecha en la leyenda (solo con 2 fechas)
    sort_by="A",             # 'A' | 'B': por qué fecha se ordenan los tickers (solo con 2 fechas)
    dual_style="auto",       # 'auto' | 'paired' | 'bullet' (solo con 2 fechas)
    compare_periods=None     # ('YTD','MTD'): compara dos períodos predefinidos (sin fechas a mano)
):
    name_mapping = load_ticker_names(tickers_file) if tickers_file else {}

    now = datetime.now()
    end_date = now + timedelta(days=1)
    start_date = None
    target_ref_date = None
    custom_label = None

    # Con dos fechas de inicio se compara cada una contra el mismo último precio.
    # (La 2.ª fecha sola, sin una 1.ª fecha personalizada, no tiene sentido y se ignora.)
    quick = bool(compare_periods) and not custom_start_date
    if quick:
        compare_periods = tuple(compare_periods)
        if len(compare_periods) != 2 or compare_periods[0] == compare_periods[1] \
                or any(p not in QUICK_PERIODS for p in compare_periods):
            print(f"Warning: comparación rápida inválida: {compare_periods}")
            return None
    dual = bool(custom_start_date and custom_start_date_2) or quick
    if custom_start_date_2 and not custom_start_date and not quick:
        print("Warning: la 2.ª fecha de inicio se ignora si no hay una 1.ª fecha de inicio personalizada.")

    if quick:
        specs = [_period_spec(pp, now) for pp in compare_periods]
        start_date = min(sp[1] for sp in specs)
    elif custom_start_date:
        start_date = custom_start_date
        if dual:   # bajar datos desde la más antigua de las dos
            start_date = min(pd.Timestamp(custom_start_date), pd.Timestamp(custom_start_date_2)).to_pydatetime()
        custom_label = f"desde {custom_start_date.strftime('%d/%m/%Y')}"
    else:
        target_ref_date, start_date = _period_spec(period, now)

    data = fetch_data(tickers, start_date, end_date)
    if data is None or data.empty:
        return None
    data = data.dropna(axis=1, how="all")
    if data.empty:
        return None

    if ccl_adjust:
        ccl_ratio = fetch_ccl_proxy(start_date, end_date)
        if ccl_ratio is not None:
            ccl_ratio_aligned = ccl_ratio.reindex(data.index).interpolate(method='time').ffill().bfill()
            data = data.div(ccl_ratio_aligned, axis=0)
            group_name += " (Ajustado CCL)"
        else:
            print("Warning: Could not fetch CCL proxy correctly.")

    # ---------- Dos fechas de inicio ----------
    if dual:
        perfs, firsts = [], []
        if quick:
            # igual que con un solo período: base = cierre del último día hábil <= fecha de referencia
            # (YTD/MTD/WTD) o primer dato desde la fecha de inicio (1Y/2Y/5Y)
            for (ref_date, st_date) in specs:
                reference_prices, base_idx = None, None
                if ref_date is not None:
                    valid = data.index[pd.Series(data.index.date <= ref_date, index=data.index)]
                    if len(valid) > 0:
                        base_idx = valid.max()
                        reference_prices = data.loc[base_idx]
                sub = data if reference_prices is not None else data.loc[data.index >= pd.Timestamp(st_date)]
                sub = sub.dropna(how='all')
                if sub.empty:
                    return None
                perfs.append(calculate_performance(sub, reference_prices))
                firsts.append(base_idx if base_idx is not None else sub.index[0])
        else:
            for d in (custom_start_date, custom_start_date_2):
                # igual que con una fecha: base = cierre del primer día hábil desde esa fecha
                sub = data.loc[data.index >= pd.Timestamp(d)].dropna(how='all')
                if sub.empty:
                    return None
                perfs.append(calculate_performance(sub))
                firsts.append(sub.index[0])
        perf_df = pd.concat([perfs[0].rename('A'), perfs[1].rename('B')], axis=1)
        perf_df = perf_df.sort_values(by='B' if str(sort_by).upper() == 'B' else 'A',
                                      ascending=False, na_position='last')
        end_str = data.index[-1].strftime('%d/%m/%Y')
        a_str, b_str = firsts[0].strftime('%d/%m/%Y'), firsts[1].strftime('%d/%m/%Y')
        # Con nombre propio, la leyenda suma la fecha real del baseline entre paréntesis
        if quick:
            series_labels = [f"{(label_a or '').strip()} ({a_str[:5]})" if (label_a or "").strip()
                             else f"{compare_periods[0]} (base: cierre {a_str})",
                             f"{(label_b or '').strip()} ({b_str[:5]})" if (label_b or "").strip()
                             else f"{compare_periods[1]} (base: cierre {b_str})"]
            period_label = f"Periodo: {compare_periods[0]} y {compare_periods[1]} hasta {end_str}"
        else:
            series_labels = [f"{(label_a or '').strip()} ({a_str[:5]})" if (label_a or "").strip() else f"Desde {a_str}",
                             f"{(label_b or '').strip()} ({b_str[:5]})" if (label_b or "").strip() else f"Desde {b_str}"]
            period_label = f"Periodo: desde {a_str} y desde {b_str} hasta {end_str}"
        return create_dual_bar_plot(
            perf_df,
            f"Performance: {group_name}",
            period_label,
            name_mapping,
            series_labels,
            subtitle=subtitle,
            style=dual_style,
            value_tags=compare_periods if quick else None
        )

    reference_prices = None
    if target_ref_date is not None and not data.empty:
        mask = pd.Series(data.index.date <= target_ref_date, index=data.index)
        valid_idx = data.index[mask]
        if len(valid_idx) > 0:
            ref_date = valid_idx.max()
            reference_prices = data.loc[ref_date]

    performance = calculate_performance(data, reference_prices)
    if performance.empty:
        return None

    last_date_fetched = data.index[-1]
    real_end_str = last_date_fetched.strftime('%d/%m/%Y')

    if custom_label:
        period_label = f"Periodo: {custom_label} - {real_end_str}"
    else:
        if reference_prices is not None:
            real_start_str = reference_prices.name.strftime('%d/%m/%Y')
        else:
            real_start_str = data.index[0].strftime('%d/%m/%Y')
        period_label = f"Periodo: {real_start_str} - {real_end_str}"

    path = create_bar_plot(
        performance,
        f"Performance: {group_name}",
        period_label,
        name_mapping,
        orientation=orientation,
        subtitle=subtitle
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', type=str, required=True)
    parser.add_argument('--group_name', type=str, default="Activos")
    parser.add_argument('--tickers_file', type=str, default=None)
    parser.add_argument('--ccl_adjust', action='store_true')
    parser.add_argument('--orientation', choices=['auto', 'horizontal', 'vertical'], default='auto')
    parser.add_argument('--subtitle', type=str, default=None, help='Aclaración opcional bajo el periodo')
    args = parser.parse_args()
    # main() is kept for backward CLI compatibility (no custom date here)
    path = run_performance(
        args.tickers,
        period=args.period,
        group_name=args.group_name,
        tickers_file=args.tickers_file,
        ccl_adjust=args.ccl_adjust,
        orientation=args.orientation,
        subtitle=args.subtitle
    )
    if path:
        print(path)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
