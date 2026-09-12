import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import logging
import os
import argparse
import sys
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
os.environ["KALEIDO_DEBUG"] = "true"

# URLs
URL_CPI_AR = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nargentina2.csv"
URL_CPI_US = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nUSA.csv"

# Elegant color for dark mode
PRIMARY_COLOR = '#40c0ff'

plot_style = {
    'plot_bgcolor': '#0a0a12',
    'paper_bgcolor': '#05050a',
    'font': dict(color='#e0e0ff', family='Arial', size=15),
    'template': 'plotly_dark',
}

# ==================== SPLITS ====================
splits = {
    'ADGO.BA': 1, 'ADBE.BA': 2, 'AEM.BA': 2, 'AMGN.BA': 3, 'AAPL.BA': 2, 'BAC.BA': 2,
    'GOLD.BA': 2, 'BIOX.BA': 2, 'CVX.BA': 2, 'LLY.BA': 7, 'XOM.BA': 2, 'FSLR.BA': 6,
    'IBM.BA': 3, 'JD.BA': 2, 'JPM.BA': 3, 'MELI.BA': 2, 'NFLX.BA': 3, 'PEP.BA': 3,
    'PFE.BA': 2, 'PG.BA': 3, 'RIO.BA': 2, 'SONY.BA': 2, 'SBUX.BA': 3, 'TXR.BA': 2,
    'BA.BA': 4, 'TM.BA': 3, 'VZ.BA': 2, 'VIST.BA': 3, 'WMT.BA': 3, 'AGRO.BA': (6, 2.1),
    'ECOG.BA': 10,
}

def ajustar_precios_por_splits(df, ticker):
    if df.empty or ticker not in splits:
        return df
    df = df.copy()
    adj = splits[ticker]
    ajustes = []

    if isinstance(adj, tuple):
        ajustes.extend([
            (datetime(2023, 11, 3), adj[0], "divide"),
            (datetime(2023, 11, 3), adj[1], "multiply"),
        ])
    else:
        fecha_split = datetime(2025, 8, 19) if ticker == "ECOG.BA" else datetime(2024, 1, 23)
        ajustes.append((fecha_split, adj, "divide"))

    for fecha, ratio, op in ajustes:
        mask = df.index < fecha
        if op == "divide":
            df.loc[mask, "Close"] /= ratio
        else:
            df.loc[mask, "Close"] *= ratio
    return df


def cargar_cpi_desde_github(url):
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date'])
        df.set_index('Date', inplace=True)
        df['Cumulative_Inflation'] = (1 + df['CPI_MoM']).cumprod()
        daily = df['Cumulative_Inflation'].resample('D').ffill().interpolate(method='linear')
        daily.index = pd.to_datetime(daily.index).tz_localize(None)
        return daily
    except Exception as e:
        logger.error(f"Error al cargar CPI desde {url}: {e}")
        return pd.Series()


daily_cpi_ar = cargar_cpi_desde_github(URL_CPI_AR)
daily_cpi_us = cargar_cpi_desde_github(URL_CPI_US)


def generate_plots(ticker, start_year=None):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    is_arg = ticker.endswith('.BA')
    start_date = f"{start_year}-01-01" if start_year else ('2020-01-01' if is_arg else '1900-01-01')
    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')

    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty or 'Close' not in data.columns:
        print(f"No hay datos para {ticker}")
        sys.exit(1)

    # Clean columns
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'][ticker].to_frame('Close')
    else:
        df = data[['Close']].copy()

    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    df = ajustar_precios_por_splits(df, ticker)

    cpi = daily_cpi_ar if is_arg else daily_cpi_us
    if cpi.empty:
        print("Error: CPI no cargado correctamente")
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

    # Elegant thin line style
    line_style = dict(
        color=PRIMARY_COLOR,
        width=1.12,
        shape='spline',
        smoothing=1.0
    )

    # Main adjusted line - Linear
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Inflation_Adjusted_Close'],
            mode='lines',
            name='Adjusted',
            line=line_style,
            hovertemplate='%{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )

    # Main adjusted line - Log
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Inflation_Adjusted_Close'],
            mode='lines',
            name='Adjusted',
            line=line_style,
            hovertemplate='%{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Subtle nominal price (ghost line)
    nominal_opacity = 0.28
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Nominal',
            line=dict(color='#778899', width=0.75, dash='dot'),
            opacity=nominal_opacity,
            hovertemplate='Nominal: %{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Nominal',
            line=dict(color='#778899', width=0.75, dash='dot'),
            opacity=nominal_opacity,
            hovertemplate='Nominal: %{y:,.2f}<extra></extra>',
            showlegend=False
        ),
        row=1, col=2
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} – Precio Ajustado por Inflación",
            font=dict(size=28),
            x=0.5,
            xanchor='center',
            y=0.96
        ),
        margin=dict(l=50, r=50, t=110, b=70),
        height=800,
        width=1800,
        hovermode='x unified',
        **plot_style
    )

    # X axes
    fig.update_xaxes(
        gridcolor='rgba(90,90,120,0.25)',
        gridwidth=0.6,
        tickformat="%b %Y",
        showgrid=True,
        row=1, col=1
    )
    fig.update_xaxes(
        gridcolor='rgba(90,90,120,0.25)',
        gridwidth=0.6,
        tickformat="%b %Y",
        showgrid=True,
        row=1, col=2
    )

    # Y axes
    fig.update_yaxes(
        title=dict(text="Precio Real", font=dict(color='#d0d0ff', size=14)),
        gridcolor='rgba(90,90,120,0.25)',
        gridwidth=0.6,
        tickfont=dict(color='#d0d0ff'),
        row=1, col=1
    )

    fig.update_yaxes(
        type="log",
        title=dict(text="Precio (Log)", font=dict(color='#d0d0ff', size=14)),
        gridcolor='rgba(90,90,120,0.25)',
        gridwidth=0.6,
        tickfont=dict(color='#d0d0ff'),
        row=1, col=2
    )

    # Watermark
    fig.add_annotation(
        text="MTaurus – X: @MTaurus_ok",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=64, color="rgba(140,140,180,0.10)"),
        textangle=0
    )

    # Save
    filename = f"inflation_combined_{ticker.replace('.', '_')}.png"
    filepath = os.path.join(output_dir, filename)
    fig.write_image(filepath, scale=4)
    print(f"✅ Saved: {filepath}")


def run_inflation_adjusted(ticker, start_year=None):
    return generate_plots(ticker, start_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument('--start_year', type=int, default=None)
    args = parser.parse_args()
    generate_plots(args.ticker, args.start_year)
