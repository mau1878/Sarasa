import argparse
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mtaurus_style  # noqa: F401  (tipografía compartida)
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
import os
import json
from datetime import datetime, timedelta
import warnings

# Ignorar warnings de yfinance
warnings.simplefilter(action='ignore', category=FutureWarning)

# ===================== CONFIG =====================
# Tema Taurus (misma paleta que heatmap_script / drawdown_plot / returns_evolution_plot)
BG_COLOR = '#0f0f1a'      # fondo de ejes
PAPER_COLOR = '#05050f'   # fondo de figura y savefig
TEXT_COLOR = '#e0e0ff'
TICK_COLOR = '#c0c0ff'
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.25)
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)
ZERO_LINE_COLOR = '#ff4d4d'
TICKER_FONT_BOOST = 3   # puntos extra del ticker respecto del valor (subir/bajar a gusto)
LABEL_GAP_PT = 5        # separación entre ticker y valor
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

def create_bar_plot(performance, title, period_label, name_mapping, filename="performance_chart.png"):
    if performance.empty:
        return None
    df = performance.to_frame(name='Return')
    # Etiqueta en una sola línea: "TICKER  valor%" (sin saltos de línea)
    df['Display Name'] = [
        " ".join(str(name_mapping.get(t, t)).split()) for t in df.index
    ]

    # Mismo divergente apagado del heatmap: rojo - gris azulado - verde
    colors = [(0.7, 0.15, 0.15), (0.3, 0.3, 0.35), (0.15, 0.6, 0.25)]
    cmap = LinearSegmentedColormap.from_list("custom_div_dark", colors, N=256)
    vmin = df['Return'].min()
    vmax = df['Return'].max()
    if vmin >= 0: vmin = -0.01
    if vmax <= 0: vmax = 0.01
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    palette_colors = [cmap(norm(v)) for v in df['Return']]

    num_bars = len(df)
    calculated_width = 10 + (num_bars * 0.5)
    fig_width = max(12, min(22, calculated_width))
    fig_height = 10

    # Tamaño de la etiqueta (ticker + valor) según cantidad de barras
    if num_bars > 50:
        label_fs = 10
    elif num_bars > 40:
        label_fs = 11
    elif num_bars > 25:
        label_fs = 12
    else:
        label_fs = 13

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)
    xs = list(range(num_bars))

    # Franjas alternas muy tenues: separan una columna de la siguiente sin líneas
    for x in range(0, num_bars, 2):
        ax.axvspan(x - 0.5, x + 0.5, color=TICK_COLOR, alpha=0.04, linewidth=0, zorder=0)
    bars = ax.bar(xs, df['Return'], color=palette_colors, width=0.8, zorder=3)

    # Etiquetas pegadas al extremo de cada barra (arriba si >=0, abajo si <0).
    # El ticker va más grande que el valor; se leen de abajo hacia arriba: "TICKER  valor%".
    ticker_fs = label_fs + TICKER_FONT_BOOST
    pairs = []  # (ticker_ann, value_ann, positive)
    for x, name, value in zip(xs, df['Display Name'], df['Return']):
        positive = value >= 0
        sign = 1 if positive else -1
        common = dict(
            xy=(x, value), textcoords='offset points', ha='center',
            va='bottom' if positive else 'top', rotation=90,
            color=TEXT_COLOR, fontweight='bold', zorder=5, annotation_clip=False,
        )
        t_ann = ax.annotate(name, xytext=(0, 4 * sign), fontsize=ticker_fs, **common)
        v_ann = ax.annotate(f'{value:.1f}%', xytext=(0, 4 * sign), fontsize=label_fs, **common)
        pairs.append((t_ann, v_ann, positive))

    ax.axhline(0, color=ZERO_LINE_COLOR, linewidth=1.5, alpha=0.8, linestyle='--', zorder=4)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y', color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)

    ax.set_xlim(-0.7, num_bars - 0.3)
    ax.set_xticks([])  # el ticker ya está en la etiqueta de cada barra

    fig.suptitle(f"{title}\n{period_label}", fontsize=22, color=TEXT_COLOR, fontweight='bold', y=0.99)
    ax.set_ylabel("Retorno (%)", fontsize=13, color=TEXT_COLOR, labelpad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    ax.tick_params(axis='y', colors=TICK_COLOR, labelsize=12)

    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(TEXT_COLOR)
        ax.spines[side].set_linewidth(0.8)

    fig.text(0.5, 0.5, '@MTaurus_ok', fontsize=90, color=WATERMARK_COLOR,
             ha='center', va='center', rotation=30, zorder=999)

    plt.tight_layout(rect=[0, 0, 1, 0.965])

    # Ajustar el eje Y midiendo el espacio real que ocupan las etiquetas
    # (están en puntos, no en unidades de datos, así que se itera hasta converger)
    # 1) Apilar ticker y valor según el largo real de cada texto (en puntos)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_pt = 72.0 / fig.dpi
    for t_ann, v_ann, positive in pairs:
        t_len = t_ann.get_window_extent(renderer).height * px_to_pt
        v_len = v_ann.get_window_extent(renderer).height * px_to_pt
        if positive:   # ticker junto a la barra, valor encima
            v_ann.xyann = (0, 4 + t_len + LABEL_GAP_PT)
        else:          # valor junto a la barra, ticker más abajo
            t_ann.xyann = (0, -(4 + v_len + LABEL_GAP_PT))

    # 2) Ajustar el eje Y a ese espacio
    y_pad = (vmax - vmin) * 0.03
    ax.set_ylim(vmin - y_pad, vmax + y_pad)
    for _ in range(4):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        inv = ax.transData.inverted()
        top, bottom = vmax, vmin
        for t_ann, v_ann, positive in pairs:
            for ann in (t_ann, v_ann):
                bb = ann.get_window_extent(renderer)
                y_lo = inv.transform((bb.x0, bb.y0))[1]
                y_hi = inv.transform((bb.x0, bb.y1))[1]
                if positive:
                    top = max(top, y_hi)
                else:
                    bottom = min(bottom, y_lo)
        span = top - bottom
        ax.set_ylim(bottom - span * 0.02, top + span * 0.02)

    output_path = os.path.abspath(filename)
    plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close()
    return output_path


# ===================== UPDATED RUNNER WITH CUSTOM START DATE =====================
def run_performance(
    tickers,
    period="YTD",
    group_name="Activos",
    tickers_file=None,
    ccl_adjust=False,
    custom_start_date=None  # ← NEW
):
    name_mapping = load_ticker_names(tickers_file) if tickers_file else {}

    now = datetime.now()
    end_date = now + timedelta(days=1)
    start_date = None
    target_ref_date = None
    custom_label = None

    if custom_start_date:
        start_date = custom_start_date
        custom_label = f"desde {custom_start_date.strftime('%d/%m/%Y')}"
    else:
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
        name_mapping
    )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', type=str, required=True)
    parser.add_argument('--group_name', type=str, default="Activos")
    parser.add_argument('--tickers_file', type=str, default=None)
    parser.add_argument('--ccl_adjust', action='store_true')
    args = parser.parse_args()
    # main() is kept for backward CLI compatibility (no custom date here)
    path = run_performance(
        args.tickers,
        period=args.period,
        group_name=args.group_name,
        tickers_file=args.tickers_file,
        ccl_adjust=args.ccl_adjust
    )
    if path:
        print(path)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
