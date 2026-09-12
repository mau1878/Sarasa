import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import os
import argparse
import sys
import warnings

import data_sources

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PRIMARY_COLOR = '#40c0ff'

plot_style = {
    'plot_bgcolor': '#0a0a12',
    'paper_bgcolor': '#05050a',
    'font': dict(color='#e0e0ff', family='Arial', size=15),
    'template': 'plotly_dark',
}


def generate_plots(ticker, start_year=None, source='yfinance'):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    is_arg = ticker.endswith('.BA')
    start_date = datetime(start_year, 1, 1).date() if start_year else (
        datetime(2020, 1, 1).date() if is_arg else datetime(1900, 1, 1).date()
    )
    end_date = (datetime.now() + timedelta(days=2)).date()

    price_df = data_sources.fetch_price_series(ticker, start_date, end_date, source=source)
    if price_df.empty:
        print(f"No hay datos para {ticker}")
        sys.exit(1)

    # data_sources.fetch_price_series ya aplica el ajuste por splits para
    # todas las fuentes, así que acá no hace falta reaplicarlo.
    df = price_df.rename(columns={data_sources._var_name(ticker): 'Close'})

    cpi = data_sources.cargar_cpi('AR' if is_arg else 'US')
    if cpi.empty:
        print("Error: CPI no cargado correctamente (ni remoto ni snapshot local)")
        sys.exit(1)

    df = df.join(cpi.to_frame('Cumulative_Inflation'), how='left')
    df['Cumulative_Inflation'] = df['Cumulative_Inflation'].ffill().bfill()

    last_cpi = df['Cumulative_Inflation'].iloc[-1]
    df['Inflation_Adjusted_Close'] = df['Close'] * (last_cpi / df['Cumulative_Inflation'])

    # === CREATE FIGURE ===
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
