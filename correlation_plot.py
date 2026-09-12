import argparse
import yfinance as yf
import pandas as pd
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

# 1. UPDATED PALETTE: Red (Negative) <-> Green (Positive)
# center='dark' ensures the middle values are black/dark gray,
# making white text pop and blending with the background.
DARK_DIVERGING = sns.diverging_palette(
    h_neg=10,    # Red
    h_pos=130,   # Green
    s=100,       # High saturation (neon look)
    l=55,        # Luminance
    sep=1,
    center='dark', # Important: Makes 0 correlation dark so white text is readable
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


def plot_correlation(corr_matrix, title):
    fig = plt.figure(figsize=(13, 11), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    # Heatmap
    heatmap = sns.heatmap(
        corr_matrix,
        cmap=DARK_DIVERGING,
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt='.2f',
        annot_kws={'size': 11, 'weight': 'bold', 'color': ANNOT_COLOR},
        cbar_kws={
            'label': 'Pearson Correlation',
            'shrink': 0.7,
            'pad': 0.03,
            'fraction': 0.046,
        },
        square=True,
        linewidths=0.8,
        linecolor='#1a1a2e',
        ax=ax
    )

    # 2. COLORBAR TEXT FIX
    # Access the colorbar object to change text color to light
    cbar = heatmap.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR) # Tick marks
    plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR) # Numbers
    cbar.set_label('Pearson Correlation', color=TEXT_COLOR, weight='bold')

    # Titles & labels
    ax.set_title(
        f"Matriz de Correlación: {title}\n({YEARS_BACK} años o período seleccionado)",
        fontsize=19, color=TEXT_COLOR, weight='bold', pad=25
    )

    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax.get_yticklabels(), rotation=0)

    # Axis labels
    ax.set_xlabel("Tickers", fontsize=12, color=TEXT_COLOR, labelpad=12)
    ax.set_ylabel("Tickers", fontsize=12, color=TEXT_COLOR, labelpad=12)

    # Watermark
    fig.text(
        0.5, 0.5,
        "MTaurus - X: @MTaurus_ok",
        fontsize=58,
        color=WATERMARK_COLOR,
        ha='center', va='center',
        rotation=30,
        weight='bold'
    )

    # Footer timestamp
    fig.text(
        0.98, 0.015,
        f"Generado: {datetime.now().strftime('%Y-%m-%d')}",
        ha='right', va='bottom',
        fontsize=9, color='gray', alpha=0.65
    )

    plt.tight_layout(pad=1.8)

    abs_path = os.path.abspath(OUTPUT_FILE)
    plt.savefig(abs_path, dpi=300, bbox_inches='tight', facecolor=PAPER_COLOR)
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