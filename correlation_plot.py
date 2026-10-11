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

# Con más tickers que esto, los números se omiten por defecto (no se leen en un feed/celular)
ANNOT_MAX_TICKERS = 30
# Líneas que separan los bloques (clusters)
BLOCK_LINE_COLOR = (224/255, 224/255, 255/255, 0.9)
BLOCK_LINE_WIDTH = 2.2


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


def _order_by_clustering(corr, n_blocks=None):
    """
    Reordena tickers por clustering jerárquico: los que se parecen quedan juntos.
    Devuelve (matriz_reordenada, ids_de_bloque) donde ids_de_bloque es un array con el
    bloque de cada ticker en el nuevo orden (o None si no se pidieron bloques / no hay scipy).
    n_blocks: cantidad de bloques (None = automático ~n/7, entre 3 y 10; 0 = sin bloques).
    """
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
        from scipy.spatial.distance import squareform
    except ImportError:
        return corr, None
    n = len(corr)
    c = corr.fillna(0).values
    dist = np.sqrt(np.clip(0.5 * (1 - c), 0, None))
    np.fill_diagonal(dist, 0)
    dist = (dist + dist.T) / 2
    Z = linkage(squareform(dist, checks=False), method='average')
    order = leaves_list(Z)
    ordered = corr.iloc[order, order]

    if n_blocks is None:
        n_blocks = int(np.clip(round(n / 7), 3, 10))
    if n_blocks and n_blocks > 1 and n > n_blocks:
        labels = fcluster(Z, t=n_blocks, criterion='maxclust')[order]
        return ordered, labels
    return ordered, None


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


def plot_correlation(corr_matrix, title, cluster=None, mask_upper=None,
                     annotate=None, n_blocks=None):
    """
    annotate:   mostrar los números en cada celda (default: True si hay <=30 tickers)
    n_blocks:   bloques a marcar con líneas gruesas al reordenar por similitud
                (default: automático, 0 = sin líneas)
    cluster:    reordenar por similitud (default: True si hay >12 tickers)
    mask_upper: mostrar solo el triángulo inferior (default: False, matriz cuadrada completa)
    El tamaño de la figura y de las fuentes escala con la cantidad de tickers.
    """
    n = len(corr_matrix)

    if cluster is None:
        cluster = n > 12
    blocks = None
    if cluster:
        corr_matrix, blocks = _order_by_clustering(corr_matrix, n_blocks)
    if mask_upper is None:
        mask_upper = False  # matriz cuadrada completa por defecto

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

    # Anotaciones: con muchos tickers se omiten; con 21-30, formato corto sin cero inicial
    if annotate is None:
        annotate = n <= ANNOT_MAX_TICKERS
    if not annotate:
        annot = False
        fmt = ''
    elif n > 20:
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
            'pad': 0.07,  # espacio extra para las etiquetas del costado derecho
            'fraction': 0.046,
        },
        square=True,
        linewidths=0.8 if n <= 25 else 0.3,
        linecolor='#1a1a2e',
        xticklabels=True,
        yticklabels=True,
        ax=ax
    )

    # Líneas gruesas entre bloques (clusters)
    if blocks is not None:
        for b in [i for i in range(1, n) if blocks[i] != blocks[i - 1]]:
            if mask_upper:
                ax.hlines(b, 0, b, color=BLOCK_LINE_COLOR, lw=BLOCK_LINE_WIDTH)
                ax.vlines(b, b, n, color=BLOCK_LINE_COLOR, lw=BLOCK_LINE_WIDTH)
            else:
                ax.axhline(b, color=BLOCK_LINE_COLOR, lw=BLOCK_LINE_WIDTH)
                ax.axvline(b, color=BLOCK_LINE_COLOR, lw=BLOCK_LINE_WIDTH)

    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
    cbar.set_label('Pearson Correlation', color=TEXT_COLOR, weight='bold')

    ax.set_title(
        f"Matriz de Correlación: {title}\n({YEARS_BACK} años o período seleccionado)",
        fontsize=min(19 * scale, 30), color=TEXT_COLOR, weight='bold', pad=25
    )

    # Tickers en los cuatro costados (abajo, arriba, izquierda, derecha)
    ax.tick_params(
        axis='both', colors=TEXT_COLOR, labelsize=tick_size,
        top=True, bottom=True, left=True, right=True,
        labeltop=True, labelbottom=True, labelleft=True, labelright=True,
    )
    for tick in ax.xaxis.get_major_ticks():
        if n > 25:
            for lbl in (tick.label1, tick.label2):
                lbl.set_rotation(90)
                lbl.set_ha('center')
        else:
            tick.label1.set_rotation(45)
            tick.label1.set_ha('right')
            tick.label1.set_rotation_mode('anchor')
            tick.label2.set_rotation(45)
            tick.label2.set_ha('left')
            tick.label2.set_rotation_mode('anchor')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label2.set_rotation(0)

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
