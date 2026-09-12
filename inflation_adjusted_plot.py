import os
import sys
import argparse
import logging
import warnings
from datetime import datetime, timedelta

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import data_sources

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ================= Plotly (interactivo) =================
PRIMARY_COLOR = '#40c0ff'
plot_style = {
    'plot_bgcolor': '#0a0a12',
    'paper_bgcolor': '#05050a',
    'font': dict(color='#e0e0ff', family='Arial', size=15),
    'template': 'plotly_dark',
}

# ================= Matplotlib (dark theme, consistente con heatmap_script.py) =================
BG_COLOR = '#0f0f1a'
PAPER_COLOR = '#05050f'
TEXT_COLOR = '#e0e0ff'
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.22)
WATERMARK_COLOR = (140 / 255, 140 / 255, 180 / 255, 0.09)
plt.style.use('dark_background')


def compute_inflation_adjusted_df(ticker, start_year=None, source='yfinance'):
    """
    Descarga el precio (multi-fuente, vía data_sources) y lo ajusta por
    inflación (IPC AR/US, con fallback local, vía data_sources). Devuelve un
    DataFrame con columnas 'Close' e 'Inflation_Adjusted_Close', o None si
    no hay datos suficientes.
    """
    ticker = ticker.strip().upper()
    is_arg = ticker.endswith('.BA')
    start_date = datetime(start_year, 1, 1).date() if start_year else (
        datetime(2020, 1, 1).date() if is_arg else datetime(1900, 1, 1).date()
    )
    end_date = (datetime.now() + timedelta(days=2)).date()

    price_df = data_sources.fetch_price_series(ticker, start_date, end_date, source=source)
    if price_df.empty:
        return None

    # data_sources.fetch_price_series ya aplica el ajuste por splits para
    # todas las fuentes, así que acá no hace falta reaplicarlo.
    df = price_df.rename(columns={data_sources._var_name(ticker): 'Close'})

    cpi = data_sources.cargar_cpi('AR' if is_arg else 'US')
    if cpi.empty:
        return None

    df = df.join(cpi.to_frame('Cumulative_Inflation'), how='left')
    df['Cumulative_Inflation'] = df['Cumulative_Inflation'].ffill().bfill()

    last_cpi = df['Cumulative_Inflation'].iloc[-1]
    df['Inflation_Adjusted_Close'] = df['Close'] * (last_cpi / df['Cumulative_Inflation'])
    return df


def build_plotly_figure(df, ticker):
    """Figura Plotly con lineal + log lado a lado (para Streamlit y para el
    guardado a PNG del modo CLI)."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Escala Lineal", "Escala Logarítmica"),
        horizontal_spacing=0.08
    )

    line_style = dict(color=PRIMARY_COLOR, width=1.12, shape='spline', smoothing=1.0)

    fig.add_trace(go.Scatter(x=df.index, y=df['Inflation_Adjusted_Close'], mode='lines',
                              name='Adjusted', line=line_style,
                              hovertemplate='%{y:,.2f}<extra></extra>', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Inflation_Adjusted_Close'], mode='lines',
                              name='Adjusted', line=line_style,
                              hovertemplate='%{y:,.2f}<extra></extra>', showlegend=False), row=1, col=2)

    nominal_opacity = 0.28
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Nominal',
                              line=dict(color='#778899', width=0.75, dash='dot'),
                              opacity=nominal_opacity, hovertemplate='Nominal: %{y:,.2f}<extra></extra>',
                              showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Nominal',
                              line=dict(color='#778899', width=0.75, dash='dot'),
                              opacity=nominal_opacity, hovertemplate='Nominal: %{y:,.2f}<extra></extra>',
                              showlegend=False), row=1, col=2)

    fig.update_layout(
        title=dict(text=f"{ticker} – Precio Ajustado por Inflación", font=dict(size=28),
                   x=0.5, xanchor='center', y=0.96),
        margin=dict(l=50, r=50, t=110, b=70),
        height=800, width=1800, hovermode='x unified',
        **plot_style
    )
    fig.update_xaxes(gridcolor='rgba(90,90,120,0.25)', gridwidth=0.6, tickformat="%b %Y", showgrid=True, row=1, col=1)
    fig.update_xaxes(gridcolor='rgba(90,90,120,0.25)', gridwidth=0.6, tickformat="%b %Y", showgrid=True, row=1, col=2)
    fig.update_yaxes(title=dict(text="Precio Real", font=dict(color='#d0d0ff', size=14)),
                      gridcolor='rgba(90,90,120,0.25)', gridwidth=0.6, tickfont=dict(color='#d0d0ff'), row=1, col=1)
    fig.update_yaxes(type="log", title=dict(text="Precio (Log)", font=dict(color='#d0d0ff', size=14)),
                      gridcolor='rgba(90,90,120,0.25)', gridwidth=0.6, tickfont=dict(color='#d0d0ff'), row=1, col=2)
    fig.add_annotation(text="MTaurus – X: @MTaurus_ok", xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=64, color="rgba(140,140,180,0.10)"), textangle=0)
    return fig


def _plot_single_matplotlib(df, ticker, scale, filepath):
    """Un solo PNG matplotlib (lineal o log), con la línea ajustada + la
    nominal fantasma de fondo — mismo estilo dark del resto de la app."""
    fig, ax = plt.subplots(figsize=(12, 7), facecolor=PAPER_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(df.index, df['Close'], color='#778899', linewidth=0.8, linestyle=':',
            alpha=0.35, label='Nominal', zorder=2)
    ax.plot(df.index, df['Inflation_Adjusted_Close'], color=PRIMARY_COLOR, linewidth=1.6,
            label='Ajustado por inflación', zorder=3)

    if scale == 'log':
        ax.set_yscale('log')
        scale_label = "Escala Logarítmica"
    else:
        scale_label = "Escala Lineal"

    ax.set_title(f"{ticker} – Precio Ajustado por Inflación\n({scale_label})",
                 fontsize=18, color=TEXT_COLOR, fontweight='bold', pad=18)
    ax.set_xlabel("Fecha", fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax.set_ylabel("Precio Real" if scale != 'log' else "Precio (Log)",
                  fontsize=12, color=TEXT_COLOR, labelpad=10)
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='both', colors=TEXT_COLOR, labelsize=10)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(TEXT_COLOR)
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.3)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=44, color=WATERMARK_COLOR,
              ha='center', va='center', rotation=30, fontweight='bold')
    fig.text(0.98, 0.015, f"Generado: {datetime.now().strftime('%Y-%m-%d')}",
              ha='right', fontsize=9, color='gray', alpha=0.65)

    plt.tight_layout()
    plt.savefig(filepath, dpi=170, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)


def generate_matplotlib_plots(ticker, start_year=None, source='yfinance'):
    """
    Genera DOS PNG separados (uno lineal, uno log) con matplotlib — más
    livianos y prácticos para ver desde el celular que el combinado Plotly.
    Devuelve (path_lineal, path_log) o (None, None) si no hay datos.
    """
    ticker = ticker.strip().upper()
    df = compute_inflation_adjusted_df(ticker, start_year, source)
    if df is None or df.empty:
        return None, None

    os.makedirs("output", exist_ok=True)
    safe_ticker = ticker.replace('.', '_')
    path_lineal = os.path.join("output", f"inflation_lineal_{safe_ticker}.png")
    path_log = os.path.join("output", f"inflation_log_{safe_ticker}.png")

    _plot_single_matplotlib(df, ticker, 'lineal', path_lineal)
    _plot_single_matplotlib(df, ticker, 'log', path_log)
    return path_lineal, path_log


def generate_plots(ticker, start_year=None, source='yfinance'):
    """Modo CLI (usado por FullManualPost.py): genera el combinado Plotly y
    lo guarda como un único PNG."""
    ticker = ticker.strip().upper()
    df = compute_inflation_adjusted_df(ticker, start_year, source)
    if df is None or df.empty:
        print(f"No hay datos para {ticker}")
        sys.exit(1)

    fig = build_plotly_figure(df, ticker)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"inflation_combined_{ticker.replace('.', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    fig.write_image(filepath, scale=4)
    print(f"|{os.path.abspath(filepath)}|")
    return filepath


def run_inflation_adjusted(ticker, start_year=None, source='yfinance'):
    return generate_plots(ticker, start_year, source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start_year', type=int, default=None)
    parser.add_argument('--source', type=str, default='yfinance')
    args = parser.parse_args()
    generate_plots(args.ticker, args.start_year, args.source)
