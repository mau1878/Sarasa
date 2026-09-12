import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from datetime import datetime, timedelta
import logging
import os
import warnings

import data_sources

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= DARK THEME CONFIG =================
BG_COLOR = '#0f0f1a'
PAPER_COLOR = '#05050f'
TEXT_COLOR = '#e0e0ff'
TICK_COLOR = '#c0c0ff'
ANNOT_COLOR = '#ffffff'
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)


def get_custom_cmap_dark():
    """Diverging colormap principal (mes/trimestre)."""
    colors = [
        (0.7, 0.15, 0.15),   # deep red
        (0.3, 0.3, 0.35),    # near neutral dark
        (0.15, 0.6, 0.25)    # deep green
    ]
    return LinearSegmentedColormap.from_list('custom_div_dark', colors, N=256)


def get_yearly_cmap_dark():
    """Mismo espíritu rojo/verde, más oscuro/saturado, para diferenciar a
    simple vista la columna de cambio anual del resto del heatmap."""
    colors = [
        (0.5, 0.0, 0.0),
        (0.3, 0.3, 0.35),
        (0.05, 0.3, 0.1)
    ]
    return LinearSegmentedColormap.from_list('yearly_div_dark', colors, N=256)


def plot_heatmap_on_axis(ax, pivot_table, analysis_period, period_label, cmap,
                          yearly_series=None):
    """Dibuja un heatmap en el axis dado. Si se pasa `yearly_series` (Serie
    indexada por año con el % de cambio anual), agrega una columna 'Año' con
    su propia escala de color, separada por una columna en blanco."""

    if analysis_period == "Mes a Mes":
        period_names = {
            1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
            7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
        }
        show_yticks = True
    else:
        period_names = {1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'}
        show_yticks = False

    num_years = len(pivot_table.index)

    # ================= DYNAMIC FONT SIZE =================
    if num_years <= 15:
        base_size = 11.5
    elif num_years <= 25:
        base_size = 10.0
    elif num_years <= 35:
        base_size = 8.8
    else:
        base_size = 7.5
    font_size = max(6.0, base_size - (num_years - 20) * 0.085)
    font_size = min(12.0, font_size)

    n_period_cols = pivot_table.shape[1]
    x_labels = [period_names.get(int(x), str(x)) for x in pivot_table.columns]

    if yearly_series is not None and not yearly_series.empty:
        combined = pivot_table.copy()
        combined['__gap__'] = np.nan
        combined['Año'] = yearly_series.reindex(pivot_table.index)
        gap_idx, year_idx = n_period_cols, n_period_cols + 1

        mask_main = np.zeros(combined.shape, dtype=bool)
        mask_main[:, gap_idx] = True
        mask_main[:, year_idx] = True

        mask_year = np.ones(combined.shape, dtype=bool)
        mask_year[:, year_idx] = combined['Año'].isna().values

        period_vals = pivot_table.values[~np.isnan(pivot_table.values)]
        p_vmin = min(-100, period_vals.min()) if period_vals.size else -100
        p_vmax = max(period_vals.max(), 1) if period_vals.size else 100
        period_norm = TwoSlopeNorm(vmin=p_vmin, vcenter=0, vmax=p_vmax)

        year_vals = combined['Año'].dropna()
        y_vmin = min(-100, year_vals.min()) if not year_vals.empty else -100
        y_vmax = max(year_vals.max(), 1) if not year_vals.empty else 100
        year_norm = TwoSlopeNorm(vmin=y_vmin, vcenter=0, vmax=y_vmax)

        sns.heatmap(combined, mask=mask_main, cmap=cmap, annot=True, fmt=".1f",
                    norm=period_norm, linewidths=0.6, linecolor='#1a1a2e', ax=ax,
                    cbar=False, annot_kws={"size": font_size, "color": ANNOT_COLOR, "weight": "bold"},
                    yticklabels=show_yticks, xticklabels=True)
        sns.heatmap(combined, mask=mask_year, cmap=get_yearly_cmap_dark(), annot=True, fmt=".1f",
                    norm=year_norm, linewidths=0.6, linecolor='#1a1a2e', ax=ax,
                    cbar=False, annot_kws={"size": font_size, "color": ANNOT_COLOR, "weight": "bold"},
                    yticklabels=show_yticks, xticklabels=True)

        # Repetir el año dentro de la columna en blanco (para no mirar hasta la izquierda)
        for i, year in enumerate(combined.index):
            ax.text(gap_idx + 0.5, i + 0.5, str(year), color=TEXT_COLOR, ha='center',
                    va='center', fontsize=max(6.0, font_size - 2), fontweight='bold')

        ax.set_xticklabels(x_labels + ['', 'Año'], rotation=45, ha='right', color=TICK_COLOR)
    else:
        sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt=".1f", center=0,
                    linewidths=0.6, linecolor='#1a1a2e', ax=ax, cbar=False,
                    annot_kws={"size": font_size, "color": ANNOT_COLOR, "weight": "bold"},
                    yticklabels=show_yticks, xticklabels=True)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', color=TICK_COLOR)

    ax.set_title(f"Variación {period_label}", fontsize=15, color=TEXT_COLOR, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Año" if show_yticks else "")
    ax.tick_params(axis='y', colors=TICK_COLOR, labelsize=10)
    ax.tick_params(axis='x', colors=TICK_COLOR, labelsize=10)
    if not show_yticks:
        ax.tick_params(axis='y', which='both', length=0)
    ax.margins(y=0.01)


def run_heatmap(ticker, start_year=None, source='yfinance', second_ticker=None,
                 third_ticker=None, apply_ccl=False):
    """
    Wrapper para Streamlit.

    source: 'yfinance' | 'analisistecnico' | 'iol' | 'byma' | 'stooq' | 'auto'
    second_ticker / third_ticker: divisores opcionales (ratio)
    apply_ccl: si True, dolariza el ticker principal antes de todo lo demás
               (usa YPFD.BA/YPF con empalme histórico BCRA para ^MERV cuando
               source='yfinance'; GD30/GD30C para las demás fuentes)
    """
    ticker = ticker.strip().upper()
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    start_date = datetime(start_year, 1, 1).date() if start_year else datetime(1990, 1, 1).date()
    end_date = (datetime.now() + timedelta(days=2)).date()

    main_df = data_sources.fetch_price_series(ticker, start_date, end_date, source=source)
    if main_df.empty:
        return None

    price_series = data_sources.apply_ratio_and_ccl(
        main_df, ticker,
        second_ticker=second_ticker, third_ticker=third_ticker,
        apply_ccl=apply_ccl, source=source, start=start_date, end=end_date,
    )
    if price_series.empty:
        return None

    df = price_series.to_frame(name='Price')

    monthly_data = df[['Price']].resample('ME').last()
    monthly_data["Cambio Mensual (%)"] = monthly_data['Price'].pct_change() * 100

    quarterly_data = df[['Price']].resample('QE').last()
    quarterly_data["Cambio Trimestral (%)"] = quarterly_data['Price'].pct_change() * 100

    yearly_changes = data_sources.compute_yearly_changes(price_series)

    years = monthly_data.index.year.unique()
    num_years = len(years)
    fig_height = max(8, 0.38 * num_years)

    fig = plt.figure(figsize=(16, fig_height), facecolor=PAPER_COLOR)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.06)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax1.set_facecolor(BG_COLOR)
    ax2.set_facecolor(BG_COLOR)
    cmap = get_custom_cmap_dark()

    monthly_pivot = monthly_data.pivot_table(
        values="Cambio Mensual (%)", index=monthly_data.index.year,
        columns=monthly_data.index.month, aggfunc="mean"
    )
    quarterly_pivot = quarterly_data.pivot_table(
        values="Cambio Trimestral (%)", index=quarterly_data.index.year,
        columns=quarterly_data.index.quarter, aggfunc="mean"
    )

    all_years = sorted(set(monthly_pivot.index) | set(quarterly_pivot.index))
    monthly_pivot = monthly_pivot.reindex(all_years)
    quarterly_pivot = quarterly_pivot.reindex(all_years)

    plot_heatmap_on_axis(ax1, monthly_pivot, "Mes a Mes", "Mensual", cmap,
                          yearly_series=yearly_changes)
    plot_heatmap_on_axis(ax2, quarterly_pivot, "Trimestre a Trimestre", "Trimestral", cmap,
                          yearly_series=yearly_changes)
    ax2.set_ylim(ax1.get_ylim())

    subtitle_parts = []
    if second_ticker:
        subtitle_parts.append(f"/ {second_ticker}")
    if third_ticker:
        subtitle_parts.append(f"/ {third_ticker}")
    if apply_ccl:
        subtitle_parts.append("(CCL)")
    subtitle = " ".join(subtitle_parts)
    display_title = f"{ticker} {subtitle}".strip()

    fig.suptitle(f"Rendimientos Históricos: {display_title}", fontsize=20,
                 color=TEXT_COLOR, fontweight="bold", y=0.99 if num_years < 12 else 1.01)

    fig.text(0.5, 0.5, "MTaurus\nX: @MTaurus_ok", ha='center', va='center',
              fontsize=42, color=WATERMARK_COLOR, rotation=45, alpha=0.085,
              fontweight='bold', linespacing=0.75)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    filename = f"combined_heatmap_{ticker.replace('.', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return filepath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', type=str, required=True)
    parser.add_argument('--start_year', type=int, default=None)
    parser.add_argument('--source', type=str, default='yfinance')
    parser.add_argument('--second_ticker', type=str, default=None)
    parser.add_argument('--third_ticker', type=str, default=None)
    parser.add_argument('--ccl_adjust', action='store_true')
    args = parser.parse_args()
    run_heatmap(args.ticker, args.start_year, args.source,
                args.second_ticker, args.third_ticker, args.ccl_adjust)
