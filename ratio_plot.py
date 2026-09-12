import argparse
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import os
import requests
import sys
from datetime import datetime, timedelta


# ===================== DATA FETCHING =====================

def fetch_data_yfinance(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
        if data.empty: return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data[['Adj Close', 'Volume']]
    except:
        return pd.DataFrame()


def fetch_data_iol(ticker, start_date, end_date):
    try:
        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com/titulo/cotizacion/BCBA/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        params = {
            'symbolName': ticker.replace('.BA', ''),
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok' and 'bars' in data:
                df = pd.DataFrame(data['bars'])
                df['Date'] = pd.to_datetime(df['time'], unit='s').dt.normalize()
                df = df.drop('time', axis=1)
                df = df.groupby('Date', as_index=False).agg({'close': 'last', 'volume': 'sum'})
                df = df.rename(columns={'close': 'Adj Close', 'volume': 'Volume'})
                df = df.set_index('Date').sort_index()
                df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
                df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
                full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                df = df.reindex(full_idx).fillna(method='ffill')
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching IOL data for {ticker}: {e}")
        return pd.DataFrame()


# ===================== PLOTTING =====================
def create_ratio_plot(main_ticker, compare_ticker, source, start_year=None):
    end_date = datetime.now()
    if start_year:
        start_date = datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365 * 5)  # Default 5 años

    # Fetch Data
    if source.lower() == 'iol':
        df_main = fetch_data_iol(main_ticker, start_date, end_date)
        df_comp = fetch_data_iol(compare_ticker, start_date, end_date)
    else:
        df_main = fetch_data_yfinance(main_ticker, start_date, end_date)
        df_comp = fetch_data_yfinance(compare_ticker, start_date, end_date)

    if df_main.empty or df_comp.empty:
        print(f"Empty data: main={df_main.shape if not df_main.empty else 'empty'}, "
              f"comp={df_comp.shape if not df_comp.empty else 'empty'}")
        return None

    # Align Data
    common_index = df_main.index.intersection(df_comp.index)
    if len(common_index) < 50:
        print(f"Too few common dates: {len(common_index)}")
        return None

    main_series = df_main.loc[common_index, 'Adj Close']
    comp_series = df_comp.loc[common_index, 'Adj Close']

    # Calculate Ratio
    ratio = main_series / comp_series

    # Stats
    ratio_mean = ratio.mean()
    ratio_std = ratio.std()
    rolling_mean = ratio.rolling(window=20).mean()
    rolling_std = ratio.rolling(window=20).std()
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    upper_stat = ratio_mean + (2 * ratio_std)
    lower_stat = ratio_mean - (2 * ratio_std)

    # Figure
    fig = go.Figure()

    # Ratio & Bands
    fig.add_trace(go.Scatter(
        x=ratio.index,
        y=ratio,
        mode='lines',
        name=f'Ratio {main_ticker}/{compare_ticker}',
        line=dict(color='#40c0ff', width=2.2)
    ))
    fig.add_trace(go.Scatter(
        x=upper_band.index,
        y=upper_band,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=lower_band.index,
        y=lower_band,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(64, 192, 255, 0.12)',
        line=dict(width=0),
        showlegend=False,
        name='Bollinger Band'
    ))

    # Static Stats
    fig.add_trace(go.Scatter(
        x=[ratio.index[0], ratio.index[-1]],
        y=[ratio_mean, ratio_mean],
        mode='lines',
        name='Mean',
        line=dict(color='#a0a0ff', dash='dash', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=[ratio.index[0], ratio.index[-1]],
        y=[upper_stat, upper_stat],
        mode='lines',
        name='+2 Std',
        line=dict(color='#ff7070', dash='dot', width=1.4)
    ))
    fig.add_trace(go.Scatter(
        x=[ratio.index[0], ratio.index[-1]],
        y=[lower_stat, lower_stat],
        mode='lines',
        name='-2 Std',
        line=dict(color='#60d070', dash='dot', width=1.4)
    ))

    # Prices (Secondary Axis)
    for series, name, color in [
        (main_series, main_ticker, '#ffaa60'),
        (comp_series, compare_ticker, '#c080ff')
    ]:
        fig.add_trace(go.Scatter(
            x=series.index,
            y=series,
            mode='lines',
            name=name,
            opacity=0.45,
            yaxis='y2',
            line=dict(color=color, width=1.6)
        ))

    # Layout - Dark theme (corrected axis titles)
    fig.update_layout(
        title=f'Ratio Analysis: {main_ticker} vs {compare_ticker}',
        yaxis=dict(
            title=dict(
                text='Ratio',
                font=dict(color='#d0d0ff')
            ),
            showgrid=True,
            gridcolor='rgba(120,120,140,0.28)',
            zerolinecolor='rgba(180,180,200,0.18)',
            tickfont=dict(color='#d0d0ff')
        ),
        yaxis2=dict(
            title=dict(
                text='Price (Log)',
                font=dict(color='#d0d0ff')
            ),
            overlaying='y',
            side='right',
            type='log',
            showgrid=False,
            tickfont=dict(color='#d0d0ff')
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(35,35,55,0.65)',
            bordercolor='rgba(90,90,130,0.45)',
            borderwidth=1
        ),
        template="plotly_dark",
        width=1600,
        height=800,
        plot_bgcolor='#0f0f1a',
        paper_bgcolor='#05050f',
        font=dict(color="#e0e0ff")
    )

    # Watermark
    fig.add_annotation(
        text="MTaurus - X: @mtaurus_ok",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=42, color="rgba(140,140,180,0.14)"),
        textangle=-30
    )

    # Output Folder Logic
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"ratio_{main_ticker}_{compare_ticker}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        fig.write_image(filepath, scale=4)
        print(f"|{os.path.abspath(filepath)}|")
        return filepath
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def run_ratio(main_ticker, compare_ticker, source="yfinance", start_year=None):
    """Wrapper alias for Streamlit."""
    return create_ratio_plot(main_ticker, compare_ticker, source, start_year)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main', required=True)
    parser.add_argument('--compare', required=True)
    parser.add_argument('--source', default='yfinance')
    parser.add_argument('--start_year', type=int, default=None, help='Año de inicio (opcional, default 5 años atrás)')
    args = parser.parse_args()

    create_ratio_plot(args.main, args.compare, args.source, args.start_year)