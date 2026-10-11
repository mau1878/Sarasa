import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime, timedelta

# ================= CONFIG - DARK THEME =================
BG_COLOR = '#0f0f1a'           # Main plot background
PAPER_COLOR = '#05050f'        # Figure background
TEXT_COLOR = '#e0e0ff'         # Soft light text
GRID_COLOR = '#2a2a3a'         # Very subtle grid
ANNOT_COLOR = '#ffffff'        # Bold annotations (Pure white for max contrast)
WATERMARK_COLOR = (140/255, 140/255, 180/255, 0.11)  # RGBA tuple

# Red (Negative) <-> Green (Positive), center dark so white text is readable
DARK_DIVERGING = sns.diverging_palette(
    h_neg=10,
    h_pos=130,
    s=100,
    l=55,
    sep=1,
    center='dark',
    as_cmap=True
)

YEARS_BACK = 5
OUTPUT_FILE = "correlation_temp.png"


def get_data_and_corr(tickers, start_year=None):
    end_date = datetime.now()
    if start_year:
        start_date = datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=YEARS_BACK * 365)

    valid_data = {}
    print(f"Downloading data for: {tickers} from {start_date.date()}")

    for t in tickers:
        try:
            df = yf.download(t, start=start_date, end=end_date, auto_adjust=True, progress=False)

            if isinstance(df.columns, pd.MultiIndex):
                try:
                    series = df['Close'][t]
                except KeyError:
                    series = df['Close']
            elif 'Close' in df.columns:
                series = df['Close']
            else:
                series = df.iloc[:, 0]

            if not series.empty and len(series) > 50:
                valid_data[t] = series
            else:
                print(f"⚠️ Warning: Insufficient data for {t} (skipping)")

        except Exception as e:
            print(f"❌ Error downloading {t}: {e}")

    if len(valid_data) < 2:
        print("❌ Error: Not enough valid tickers to build correlation matrix.")
        return None

    df_all = pd.DataFrame(valid_data)
    returns = df_all.pct_change()
    corr_matrix = returns.corr(method='pearson', min_periods=30)

    return corr_matrix


def _order_by_clustering(corr):
    """Reordena tickers por clustering jerárquico: los que se parecen quedan juntos."""
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform
    except ImportError:
        return corr
    c = corr.fillna(0).values
    dist = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    order = leaves_list(linkage(squareform(dist, checks=False), method='average'))
    return corr.iloc[order, order]


def _short_fmt(v):
    """0.85 -> '.85', -0.85 -> '-.85', 1.00 -> '1' (más angosto, entra en celdas chicas)."""
    if pd.isna(v):
        return ""
    s = f"{v:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    if s in ("1.00", "-1.00"):
        return s[:-3]
    return s


def plot_correlation(corr_matrix, title, cluster=None, mask_upper=None):
    """
    cluster:    reordenar por similitud (default: True si hay >12 tickers)
    mask_upper: mostrar solo el triángulo inferior (default: True si hay >25 tickers)
    El tamaño de la figura y de las fuentes escala con la cantidad de tickers.
    """
    n = len(corr_matrix)

    if cluster is None:
        cluster = n > 12
    if cluster:
        corr_matrix = _order_by_clustering(corr_matrix)
    if mask_upper is None:
        mask_upper = n > 25

    # ---------- Tamaños adaptativos ----------
    fig_w = max(13, 0.42 * n + 3)
    fig_h = fig_w * 11 / 13
    scale = fig_w / 13
    cell_pt = fig_h * 0.78 * 72 / n                      # lado aprox. de una celda en puntos
    annot_size = float(np.clip(cell_pt * 0.34, 6, 11))
    tick_size = float(np.clip(cell_pt * 0.42, 7, 11))
    dpi = 300 if n <= 25 else 200

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    # Anotaciones: con muchos tickers, formato corto sin cero inicial
    if n > 20:
        annot = corr_matrix.apply(lambda col: col.map(_short_fmt)).values
        fmt = ''
    else:
        annot = True
        fmt = '.2f'

    mask = np.triu(np.ones((n, n), dtype=bool), k=1) if mask_upper else None

    heatmap = sns.heatmap(
        corr_matrix,
        cmap=DARK_DIVERGING,
        center=0,
        vmin=-1,
        vmax=1,
        mask=mask,
        annot=annot,
        fmt=fmt,
        annot_kws={'size': annot_size, 'weight': 'bold', 'color': ANNOT_COLOR},
        cbar_kws={
            'label': 'Pearson Correlation',
            'shrink': 0.7,
            'pad': 0.03,
            'fraction': 0.046,
        },
        square=True,
        linewidths=0.8 if n <= 25 else 0.3,
        linecolor='#1a1a2e',
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.set_label('Pearson Correlation', color=TEXT_COLOR, weight='bold')

    ax.set_title(
        f"Matriz de Correlación: {title}\n({YEARS_BACK} años o período seleccionado)",
        fontsize=min(19 * scale, 30), color=TEXT_COLOR, weight='bold', pad=25
    )

    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=tick_size)
    if n > 25:
        plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)

    ax.set_xlabel("Tickers", fontsize=12 * min(scale, 1.6), color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Tickers", fontsize=12 * min(scale, 1.6), color=TEXT_COLOR, labelpad=12)

    fig.text(
        0.5, 0.5,
        "MTaurus - X: @MTaurus_ok",
        fontsize=58 * scale,
        color=WATERMARK_COLOR,
        ha='center', va='center',
        rotation=30,
        weight='bold'
    )

    fig.text(
        0.98, 0.015,
        f"Generado: {datetime.now().strftime('%Y-%m-%d')}",
        ha='right', va='bottom',
        fontsize=9 * min(scale, 1.6), color='gray', alpha=0.65
    )

    plt.tight_layout(pad=1.8)

    abs_path = os.path.abspath(OUTPUT_FILE)
    plt.savefig(abs_path, dpi=dpi, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)

    return abs_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True, help='List of tickers (space separated)')
    parser.add_argument('--title', type=str, default="Acciones", help='Título del gráfico')
    parser.add_argument('--start_year', type=int, default=None, help='Año de inicio (opcional)')
    args = parser.parse_args()

    try:
        unique_tickers = list(dict.fromkeys([t.upper() for t in args.tickers]))
        corr_matrix = get_data_and_corr(unique_tickers, args.start_year)
        if corr_matrix is None or corr_matrix.empty:
            sys.exit(1)

        path = plot_correlation(corr_matrix, args.title)
        print(f"|{path}|")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_correlation(tickers, title="Acciones", start_year=None):
    """Wrapper for Streamlit: builds correlation heatmap."""
    unique_tickers = list(dict.fromkeys([t.upper() for t in tickers]))
    corr_matrix = get_data_and_corr(unique_tickers, start_year)
    if corr_matrix is None or corr_matrix.empty:
        return None
    path = plot_correlation(corr_matrix, title)
    return path

if __name__ == "__main__":
    main()
