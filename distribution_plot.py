"""
distribution_plot.py
=====================
Análisis de distribución de variaciones periódicas para un ticker (con soporte
de ratio/CCL igual que heatmap_script.py, vía data_sources.py):

- run_histogram: histograma de variaciones mensuales/trimestrales + ajuste
  gaussiano + percentiles (5/25/50/75/95), estilo dark consistente con el resto
  de la app.
- run_period_ranking: barras de meses/trimestres positivos vs negativos, con
  un bucket extra "Total Años (histórico)" al final.
- run_streaks: NO genera imagen; devuelve una lista de rachas (dicts) para
  mostrar con st.write/st.dataframe en la app, ya que es información tabular.

Reutiliza drawdown_plot.py para el análisis de drawdown (no se reimplementa acá).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings

import data_sources

warnings.filterwarnings("ignore")
plt.style.use('dark_background')

# ================= DARK THEME CONFIG (consistente con heatmap_script.py) =================
BG_COLOR = '#0f0f1a'
PAPER_COLOR = '#05050f'
TEXT_COLOR = '#e0e0ff'
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.22)
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)
POS_COLOR = '#40e070'
NEG_COLOR = '#ff5555'


def _fetch_series(ticker, start_year, source, second_ticker, third_ticker, apply_ccl):
    ticker = ticker.strip().upper()
    start_date = datetime(start_year, 1, 1).date() if start_year else datetime(1990, 1, 1).date()
    end_date = (datetime.now() + timedelta(days=2)).date()

    main_df = data_sources.fetch_price_series(ticker, start_date, end_date, source=source)
    if main_df.empty:
        return None, ticker

    series = data_sources.apply_ratio_and_ccl(
        main_df, ticker, second_ticker=second_ticker, third_ticker=third_ticker,
        apply_ccl=apply_ccl, source=source, start=start_date, end=end_date,
    )
    return series, ticker


def _display_title(ticker, second_ticker, third_ticker, apply_ccl):
    parts = [ticker]
    if second_ticker:
        parts.append(f"/ {second_ticker}")
    if third_ticker:
        parts.append(f"/ {third_ticker}")
    if apply_ccl:
        parts.append("(CCL)")
    return " ".join(parts)


def _add_watermark(fig):
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=46, color=WATERMARK_COLOR,
              ha='center', va='center', rotation=30, fontweight='bold')


def run_histogram(ticker, start_year=None, source='yfinance', second_ticker=None,
                   third_ticker=None, apply_ccl=False, frequency='Mensual'):
    """frequency: 'Mensual' o 'Trimestral'. Devuelve el path del PNG o None."""
    series, ticker = _fetch_series(ticker, start_year, source, second_ticker, third_ticker, apply_ccl)
    if series is None or series.empty:
        return None

    freq = 'ME' if frequency == 'Mensual' else 'QE'
    changes = data_sources.compute_period_changes(series, freq=freq).dropna()
    if changes.empty or len(changes) < 5:
        return None

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.hist(changes, bins=30, density=True, color='#40c0ff', alpha=0.75,
            edgecolor='#0f0f1a', linewidth=0.5, zorder=3)

    mu, std = norm.fit(changes)
    x = np.linspace(changes.min(), changes.max(), 200)
    ax.plot(x, norm.pdf(x, mu, std), color='white', linewidth=2.4, zorder=4,
            label=f"Gaussiana (μ={mu:.1f}%, σ={std:.1f}%)")

    percentile_colors = {5: '#ef5350', 25: '#ffb300', 50: '#4caf50', 75: '#42a5f5', 95: '#ab47bc'}
    y_top = ax.get_ylim()[1]
    for p, color in percentile_colors.items():
        val = np.percentile(changes, p)
        ax.axvline(val, color=color, linestyle='--', alpha=0.85, linewidth=1.3, zorder=5)
        ax.text(val, y_top * 0.97, f'p{p}: {val:.1f}%', color=color, fontsize=9,
                rotation=90, va='top', ha='right', fontweight='bold')

    ax.axvline(0, color=TEXT_COLOR, linewidth=1.2, alpha=0.6, zorder=2)
    ax.set_title(f"Distribución de Variaciones {frequency}es: {_display_title(ticker, second_ticker, third_ticker, apply_ccl)}",
                 fontsize=17, color=TEXT_COLOR, fontweight='bold', pad=18)
    ax.set_xlabel(f"Variación {frequency} (%)", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Densidad", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(TEXT_COLOR)
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.3)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _add_watermark(fig)
    fig.text(0.98, 0.015, f"Generado: {datetime.now().strftime('%Y-%m-%d')}",
              ha='right', fontsize=9, color='gray', alpha=0.65)

    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", f"histograma_{frequency.lower()}_{ticker.replace('.', '_')}.png")
    plt.savefig(filepath, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return filepath


def run_period_ranking(ticker, start_year=None, source='yfinance', second_ticker=None,
                        third_ticker=None, apply_ccl=False, analysis_period='Mes a Mes'):
    """analysis_period: 'Mes a Mes' o 'Trimestre a Trimestre'."""
    series, ticker = _fetch_series(ticker, start_year, source, second_ticker, third_ticker, apply_ccl)
    if series is None or series.empty:
        return None

    freq = 'ME' if analysis_period == 'Mes a Mes' else 'QE'
    changes = data_sources.compute_period_changes(series, freq=freq).dropna()
    if changes.empty:
        return None

    if analysis_period == 'Mes a Mes':
        grp = changes.index.month
        names = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    else:
        grp = changes.index.quarter
        names = ['Q1', 'Q2', 'Q3', 'Q4']

    pos = changes.groupby(grp).apply(lambda x: (x > 0).sum())
    neg = changes.groupby(grp).apply(lambda x: (x < 0).sum())
    pos = pos.reindex(range(1, len(names) + 1), fill_value=0)
    neg = neg.reindex(range(1, len(names) + 1), fill_value=0)

    yearly_changes = data_sources.compute_yearly_changes(series).dropna()
    pos_years = int((yearly_changes > 0).sum())
    neg_years = int((yearly_changes < 0).sum())

    x_periods = np.arange(len(names))
    x_year = len(names) + 1  # hueco de separación real

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.bar(x_periods - 0.2, pos.values, 0.4, label='Positivos', color=POS_COLOR, zorder=3)
    ax.bar(x_periods + 0.2, neg.values, 0.4, label='Negativos', color=NEG_COLOR, zorder=3)
    ax.bar(x_year - 0.2, pos_years, 0.4, color=POS_COLOR, zorder=3)
    ax.bar(x_year + 0.2, neg_years, 0.4, color=NEG_COLOR, zorder=3)
    ax.axvline((x_periods[-1] + x_year) / 2, color=TEXT_COLOR, linestyle=':', alpha=0.4)

    ax.set_xticks(list(x_periods) + [x_year])
    ax.set_xticklabels(names + ['Total Años\n(histórico)'], rotation=45, ha='right', color=TEXT_COLOR)
    ax.set_title(f"Ranking {'Mensual' if analysis_period == 'Mes a Mes' else 'Trimestral'}: "
                 f"{_display_title(ticker, second_ticker, third_ticker, apply_ccl)}",
                 fontsize=17, color=TEXT_COLOR, fontweight='bold', pad=18)
    ax.set_ylabel("Cantidad de períodos", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax.grid(True, axis='y', color=GRID_COLOR, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(TEXT_COLOR)
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.3)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    _add_watermark(fig)
    plt.tight_layout()
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", f"ranking_{ticker.replace('.', '_')}.png")
    plt.savefig(filepath, dpi=180, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return filepath


def run_streaks(ticker, start_year=None, source='yfinance', second_ticker=None,
                 third_ticker=None, apply_ccl=False, analysis_period='Mes a Mes'):
    """
    Devuelve una lista de dicts con las rachas de períodos consecutivos
    positivos/negativos: [{'inicio', 'fin', 'longitud', 'signo'}, ...]
    No genera imagen: es tabular, se muestra con st.dataframe/st.write.
    """
    series, ticker = _fetch_series(ticker, start_year, source, second_ticker, third_ticker, apply_ccl)
    if series is None or series.empty:
        return []

    freq = 'ME' if analysis_period == 'Mes a Mes' else 'QE'
    changes = data_sources.compute_period_changes(series, freq=freq).dropna()
    if len(changes) < 3:
        return []

    streaks = []
    current_sign, current_start, current_len = None, None, 0
    for idx, val in changes.items():
        sign = val > 0
        if current_sign is None:
            current_sign, current_start, current_len = sign, idx, 1
        elif sign == current_sign:
            current_len += 1
        else:
            streaks.append({
                'inicio': current_start, 'fin': idx, 'longitud': current_len,
                'signo': 'positiva' if current_sign else 'negativa'
            })
            current_sign, current_start, current_len = sign, idx, 1
    if current_len > 0:
        streaks.append({
            'inicio': current_start, 'fin': changes.index[-1], 'longitud': current_len,
            'signo': 'positiva' if current_sign else 'negativa'
        })
    return streaks
