"""
stock_adjuster.py
==================
Puerto de `graficar_activos_ajustados` (y funciones asociadas) del script
original `inflacion.py`: ajustadora multi-ticker por inflación con salida
Plotly (interactiva) + Matplotlib/Seaborn (liviana para celular), SMA del
primer ticker, línea fantasma nominal, modo porcentual, y anotaciones de
splits/eventos personalizados cargados por el usuario.

Requiere streamlit (para session_state: splits/eventos/CSVs personalizados),
a diferencia del resto de los módulos que son agnósticos de UI.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import plotly.graph_objs as go
import streamlit as st

import data_sources

PLOT_STYLE = {
    'plot_bgcolor': 'rgb(30, 30, 30)',
    'paper_bgcolor': 'rgb(30, 30, 30)',
    'font': dict(color='white', family='Arial', size=14),
    'xaxis': dict(gridcolor='rgba(255,255,255,0.2)', zerolinecolor='rgba(255,255,255,0.2)',
                  tickformat="%b %Y", tickangle=45, nticks=10),
    'yaxis': dict(gridcolor='rgba(255,255,255,0.2)', zerolinecolor='rgba(255,255,255,0.2)', tickformat=".2f"),
    'legend': dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                   font=dict(size=12, color='white'), bgcolor='rgba(30,30,30,0.8)',
                   bordercolor='rgba(255,255,255,0.2)', borderwidth=1),
    'template': 'plotly_dark',
    'transition_duration': 0,
    'autosize': True,
}

plt.rcParams.update({
    'figure.facecolor': '#1e1e1e', 'axes.facecolor': '#1e1e1e', 'axes.edgecolor': 'white',
    'axes.labelcolor': 'white', 'text.color': 'white', 'xtick.color': 'white',
    'ytick.color': 'white', 'grid.color': 'white', 'grid.alpha': 0.2,
    'legend.facecolor': '#1e1e1e', 'legend.edgecolor': 'white', 'legend.labelcolor': 'white',
    'font.size': 10,
})


# ===================== Splits/eventos personalizados (session_state) =====================

def init_session_state():
    if "custom_splits" not in st.session_state:
        st.session_state.custom_splits = []
    if "custom_events" not in st.session_state:
        st.session_state.custom_events = []
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = {}


def _aplicar_splits_personalizados(df, ticker, price_col='Close'):
    """Aplica, ENCIMA de los splits ya resueltos por data_sources, los splits
    personalizados que el usuario cargó a mano para este ticker."""
    if df.empty:
        return df
    df = df.copy()
    custom = [s for s in st.session_state.get("custom_splits", []) if s["ticker"] == ticker]
    for split in sorted(custom, key=lambda s: s["date"]):
        split_dt = datetime.combine(split["date"], datetime.min.time())
        df.loc[df.index <= split_dt, price_col] /= split["ratio"]
    return df


def _fetch_ticker_data(ticker, start_date, end_date, source):
    """CSV subido a mano > fetcher multi-fuente unificado."""
    ticker_upper = ticker.upper()
    uploaded = st.session_state.get("uploaded_data", {})
    if ticker_upper in uploaded:
        df = uploaded[ticker_upper]
        start_dt, end_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
        return df[(df.index >= start_dt) & (df.index <= end_dt)].rename(
            columns={c: 'Close' for c in df.columns if c == 'Close'}
        )
    price_df = data_sources.fetch_price_series(ticker_upper, start_date, end_date, source=source)
    if price_df.empty:
        return pd.DataFrame()
    df = price_df.rename(columns={data_sources._var_name(ticker_upper): 'Close'})
    return _aplicar_splits_personalizados(df, ticker_upper)


# ===================== Ajustadora multi-ticker (Plotly + Matplotlib) =====================

def _finalizar_grafico_mpl(fig, ax, titulo, ylabel_txt, is_percentage, use_log_scale):
    ax.set_title(titulo, fontsize=15, color='white', pad=12)
    ax.set_xlabel('Fecha', fontsize=11)
    ax.set_ylabel(ylabel_txt, fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    fig.autofmt_xdate(rotation=45)
    if is_percentage:
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    else:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.2f}'))
        if use_log_scale:
            ax.set_yscale('log')
    ax.text(0.5, 0.5, "MTaurus - X: mtaurus_ok", transform=ax.transAxes,
            fontsize=24, color='white', alpha=0.12, ha='center', va='center')
    fig.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ncols = min(len(labels), 3)
        filas_leyenda = -(-len(labels) // ncols)
        espacio_inferior = 0.22 + 0.06 * filas_leyenda
        fig.subplots_adjust(bottom=espacio_inferior)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -espacio_inferior * 1.35), ncol=ncols, fontsize=8)


def graficar_activos_ajustados(
    tickers_input, sma_period, plot_start_date, daily_cpi_serie, source, moneda, colors,
    is_percentage_mode, show_percentage_from_recent, use_log_scale, show_nominal_ghost,
    siempre_ajustar=False, force_inflation=False, plot_end_date=None,
):
    """
    Descarga, ajusta por inflación y grafica (Plotly + Matplotlib/Seaborn) una
    lista de tickers separados por comas. Devuelve (fig_plotly, fig_mpl,
    stock_data_dict_nominal, stock_data_dict_adjusted, ticker_var_map).
    """
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    fig = go.Figure()
    fig_mpl, ax_mpl = plt.subplots(figsize=(11, 5.5))

    ticker_var_map = {t: t.replace('.', '_') for t in tickers}
    stock_data_dict_nominal, stock_data_dict_adjusted = {}, {}

    end_date = (plot_end_date or daily_cpi_serie.index.max().date()) + timedelta(days=1)
    errors = []

    for i, ticker in enumerate(tickers):
        try:
            stock_data = _fetch_ticker_data(ticker, plot_start_date, end_date, source)
            if stock_data.empty:
                errors.append(f"No se encontraron datos para el ticker {ticker}.")
                continue

            needs_adj = True if siempre_ajustar else (
                force_inflation or ticker.endswith('.BA') or ticker == '^MERV' or source != 'yfinance'
            )
            if needs_adj:
                cpi_clean = daily_cpi_serie.copy()
                cpi_clean.index = pd.to_datetime(cpi_clean.index).normalize()
                stock_data = stock_data.join(cpi_clean.to_frame('Cumulative_Inflation'), how='left')
                stock_data['Cumulative_Inflation'] = stock_data['Cumulative_Inflation'].ffill().bfill()
                if not stock_data.empty and not stock_data['Cumulative_Inflation'].isna().all():
                    last_cpi = stock_data['Cumulative_Inflation'].iloc[-1]
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (
                        last_cpi / stock_data['Cumulative_Inflation']
                    )
                else:
                    stock_data['Inflation_Adjusted_Close'] = stock_data['Close']
            else:
                stock_data['Inflation_Adjusted_Close'] = stock_data['Close']

            var_name = ticker_var_map[ticker]
            stock_data_dict_nominal[var_name] = stock_data['Close']
            stock_data_dict_adjusted[var_name] = stock_data['Inflation_Adjusted_Close']

            display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker
            color = colors[i % len(colors)]

            if is_percentage_mode:
                if show_percentage_from_recent and len(stock_data) > 0:
                    pct = ((stock_data['Inflation_Adjusted_Close'].iloc[-1] /
                            stock_data['Inflation_Adjusted_Close']) - 1) * 100
                    pct = pct.clip(lower=-100)
                else:
                    pct = (stock_data['Inflation_Adjusted_Close'] /
                           stock_data['Inflation_Adjusted_Close'].iloc[0] - 1) * 100

                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=pct, mode='lines', name=f'{display_name} (Ajustado, %)',
                    line=dict(color=color, width=1.5),
                    hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'
                ))
                fig.add_shape(type="line", x0=stock_data.index.min(), x1=stock_data.index.max(),
                              y0=0, y1=0, line=dict(color="rgba(255,0,0,0.5)", width=1, dash="dash"))
                ax_mpl.plot(stock_data.index, pct, color=color, linewidth=1.5, label=f'{display_name} (Ajustado, %)')
                ax_mpl.axhline(0, color='red', linewidth=1, linestyle='--', alpha=0.5)

                if show_nominal_ghost:
                    if show_percentage_from_recent and len(stock_data) > 0:
                        pct_nom = ((stock_data['Close'].iloc[-1] / stock_data['Close']) - 1) * 100
                        pct_nom = pct_nom.clip(lower=-100)
                    else:
                        pct_nom = (stock_data['Close'] / stock_data['Close'].iloc[0] - 1) * 100
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, y=pct_nom, mode='lines', name=f'{display_name} Nominal (%)',
                        line=dict(color=color, width=1, dash='dot'), opacity=0.55,
                        hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación nominal: %{y:.2f}%<extra></extra>'
                    ))
                    ax_mpl.plot(stock_data.index, pct_nom, color=color, linewidth=1, linestyle=':',
                                alpha=0.55, label=f'{display_name} Nominal (%)')
            else:
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'], mode='lines',
                    name=f'{display_name} (Ajustado por Inflación)', line=dict(color=color, width=1.5),
                    hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Precio: %{{y:.2f}} {moneda}<extra></extra>'
                ))
                avg_price = stock_data['Inflation_Adjusted_Close'].mean()
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=[avg_price] * len(stock_data), mode='lines',
                    name=f'{display_name} Promedio (Ajustado)', line=dict(color=color, width=0.8, dash='dot'),
                    hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Promedio: %{{y:.2f}} {moneda}<extra></extra>'
                ))
                ax_mpl.plot(stock_data.index, stock_data['Inflation_Adjusted_Close'], color=color,
                            linewidth=1.5, label=f'{display_name} (Ajustado por Inflación)')
                ax_mpl.axhline(avg_price, color=color, linewidth=0.8, linestyle=':', alpha=0.8)

                if show_nominal_ghost:
                    fig.add_trace(go.Scatter(
                        x=stock_data.index, y=stock_data['Close'], mode='lines', name=f'{display_name} Nominal',
                        line=dict(color=color, width=1, dash='dot'), opacity=0.5,
                        hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>Nominal: %{{y:.2f}} {moneda}<extra></extra>'
                    ))
                    ax_mpl.plot(stock_data.index, stock_data['Close'], color=color, linewidth=1,
                                linestyle=':', alpha=0.5, label=f'{display_name} Nominal')

            if i == 0 and len(stock_data) > 0:
                stock_data['SMA'] = stock_data['Inflation_Adjusted_Close'].rolling(window=sma_period).mean()
                fig.add_trace(go.Scatter(
                    x=stock_data.index, y=stock_data['SMA'], mode='lines', name=f'{display_name} SMA (Ajustado)',
                    line=dict(color='orange', width=1),
                    hovertemplate=f'Fecha: %{{x|%Y-%m-%d}}<br>SMA: %{{y:.2f}} {moneda}<extra></extra>'
                ))

            for split in st.session_state.get("custom_splits", []):
                if split["ticker"] == ticker:
                    split_dt = datetime.combine(split["date"], datetime.min.time())
                    fig.add_vline(x=split_dt.timestamp() * 1000, line=dict(color="white", width=1, dash="dash"),
                                  annotation_text=f"Split {split['ratio']}:1", annotation_position="top")
                    ax_mpl.axvline(split_dt, color='white', linewidth=1, linestyle='--', alpha=0.7)

            for event in st.session_state.get("custom_events", []):
                if event["ticker"] == ticker:
                    event_dt = datetime.combine(event["date"], datetime.min.time())
                    fig.add_vline(x=event_dt.timestamp() * 1000, line=dict(color="yellow", width=1, dash="dot"),
                                  annotation_text=event["description"], annotation_position="top")
                    ax_mpl.axvline(event_dt, color='yellow', linewidth=1, linestyle=':', alpha=0.7)

        except Exception as e:
            errors.append(f"Error procesando {ticker}: {e}")
            continue

    fig.add_annotation(text="MTaurus - X: mtaurus_ok", xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=30, color="rgba(255,255,255,0.2)"), opacity=0.15)

    tickers_titulo = ', '.join(tickers) if len(tickers) <= 3 else f"{len(tickers)} tickers"
    titulo_base = f'Precios Históricos Ajustados por Inflación ({moneda}) - {tickers_titulo}'
    titulo = titulo_base if not is_percentage_mode else f'{titulo_base} (%)'
    ylabel = f'Precio de Cierre Ajustado ({moneda})' if not is_percentage_mode else 'Variación Porcentual (%)'

    fig.update_layout(title=dict(text=titulo, font=dict(size=20, color='white')),
                       xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
                       yaxis_title=dict(text=ylabel, font=dict(size=14, color='white')), **PLOT_STYLE)
    fig.update_yaxes(type='log' if use_log_scale else 'linear', tickformat=',.2f',
                      ticksuffix='' if not is_percentage_mode else '%')

    _finalizar_grafico_mpl(fig_mpl, ax_mpl, titulo, ylabel, is_percentage_mode, use_log_scale)

    return fig, fig_mpl, stock_data_dict_nominal, stock_data_dict_adjusted, ticker_var_map, errors


# ===================== Cálculos / ratios personalizados =====================

def evaluate_custom_expression(custom_expression, stock_data_dict_nominal, ticker_var_map,
                                daily_cpi_serie, colors, custom_title=None,
                                show_percentage=False, show_percentage_from_recent=False,
                                use_log_scale=False):
    """
    Evalúa una expresión matemática sobre los tickers ya cargados (p.ej.
    'META*(YPFD.BA / YPF)/20'), ajusta por inflación si interviene algún
    ticker .BA, y devuelve una figura Plotly (o levanta excepción con el
    detalle del error, igual que el script original).
    """
    import re

    sorted_tickers = sorted(ticker_var_map.keys(), key=len, reverse=True)
    transformed_expression = custom_expression
    used_tickers, used_ba_tickers = set(), set()

    for ticker in sorted_tickers:
        if ticker in custom_expression:
            used_tickers.add(ticker)
            var_name = ticker_var_map[ticker]
            pattern = re.escape(ticker)
            transformed_expression = re.sub(rf'\b{pattern}\b', var_name, transformed_expression)
            if ticker.endswith('.BA'):
                used_ba_tickers.add(ticker)

    combined_nominal_df = pd.DataFrame({
        ticker_var_map[t]: stock_data_dict_nominal[ticker_var_map[t]] for t in used_tickers
    })
    combined_nominal_df.dropna(inplace=True)
    if combined_nominal_df.empty:
        raise ValueError("No hay datos disponibles para todos los tickers seleccionados en las fechas especificadas.")

    custom_series_nominal = combined_nominal_df.eval(transformed_expression, engine='python')
    custom_series_nominal = custom_series_nominal.to_frame(name='Custom_Nominal')

    if used_ba_tickers:
        custom_series_nominal = custom_series_nominal.join(
            daily_cpi_serie.to_frame('Cumulative_Inflation'), how='inner'
        )
        custom_series_nominal['Cumulative_Inflation'] = custom_series_nominal['Cumulative_Inflation'].ffill()
        custom_series_nominal.dropna(subset=['Cumulative_Inflation'], inplace=True)
        custom_series_nominal['Inflation_Adjusted_Custom'] = custom_series_nominal['Custom_Nominal'] * (
            custom_series_nominal['Cumulative_Inflation'].iloc[-1] / custom_series_nominal['Cumulative_Inflation']
        )
        adjusted_series = custom_series_nominal['Inflation_Adjusted_Custom']
    else:
        adjusted_series = custom_series_nominal['Custom_Nominal']

    fig = go.Figure()

    for event in st.session_state.get("custom_events", []):
        if event["ticker"] in used_tickers:
            event_dt = datetime.combine(event["date"], datetime.min.time())
            fig.add_vline(x=event_dt.timestamp() * 1000, line=dict(color="yellow", width=1, dash="dot"),
                          annotation_text=event["description"], annotation_position="top left")

    for split in st.session_state.get("custom_splits", []):
        if split["ticker"] in used_tickers:
            split_dt = datetime.combine(split["date"], datetime.min.time())
            fig.add_vline(x=split_dt.timestamp() * 1000, line=dict(color="white", width=1, dash="dash"),
                          annotation_text=f"Split {split['ratio']}:1", annotation_position="top left")

    label = f'Custom: {custom_expression[:15]}...' if len(custom_expression) > 15 else custom_expression
    if show_percentage or show_percentage_from_recent:
        if show_percentage_from_recent:
            pct = -((adjusted_series / adjusted_series.iloc[-1] - 1) * 100)
        else:
            pct = (adjusted_series / adjusted_series.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(x=pct.index, y=pct, mode='lines', name=label,
                                  line=dict(color=colors[-1], width=2),
                                  hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Variación: %{y:.2f}%<extra></extra>'))
        fig.add_hline(y=0, line=dict(color="rgba(255,0,0,0.5)", dash="dash"))
    else:
        fig.add_trace(go.Scatter(x=adjusted_series.index, y=adjusted_series, mode='lines', name=label,
                                  line=dict(color=colors[-1], width=2),
                                  hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Valor: %{y:.2f} ARS<extra></extra>'))

    plot_title = custom_title.strip() if custom_title and custom_title.strip() else (
        'Ratio / Cálculo Personalizado (%)' if (show_percentage or show_percentage_from_recent)
        else 'Ratio / Cálculo Personalizado Ajustado por Inflación'
    )

    fig.update_layout(
        title=dict(text=plot_title, font=dict(size=20, color='white')),
        xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
        yaxis_title=dict(text='Variación (%)' if (show_percentage or show_percentage_from_recent) else 'Valor Ajustado (ARS)',
                          font=dict(size=14, color='white')),
        **PLOT_STYLE
    )
    fig.update_yaxes(
        type='log' if (not (show_percentage or show_percentage_from_recent) and use_log_scale) else 'linear',
        tickformat=',.2f', ticksuffix='%' if (show_percentage or show_percentage_from_recent) else ''
    )
    fig.add_annotation(text="MTaurus - X: mtaurus_ok", xref="paper", yref="paper", x=0.5, y=0.5,
                        showarrow=False, font=dict(size=30, color="rgba(255,255,255,0.15)"), opacity=0.2)
    return fig


# ===================== Volatilidad histórica ajustada por inflación =====================

def build_volatility_figure(ticker, start_date, end_date, source, daily_cpi_serie, colors, volatility_window=20):
    """Precio ajustado + volatilidad histórica (rolling std anualizada) en
    ejes duales. Devuelve (fig, latest_volatility) o (None, None)."""
    ticker = ticker.strip().upper()
    stock_data = _fetch_ticker_data(ticker, start_date, end_date, source)
    if stock_data.empty:
        return None, None

    if ticker.endswith('.BA'):
        cpi_clean = daily_cpi_serie.copy()
        cpi_clean.index = pd.to_datetime(cpi_clean.index).normalize()
        stock_data = stock_data.join(cpi_clean.to_frame('Cumulative_Inflation'), how='left')
        stock_data['Cumulative_Inflation'] = stock_data['Cumulative_Inflation'].ffill()
        stock_data = stock_data.dropna(subset=['Cumulative_Inflation'])
        if stock_data.empty:
            return None, None
        stock_data['Inflation_Adjusted_Close'] = stock_data['Close'] * (
            stock_data['Cumulative_Inflation'].iloc[-1] / stock_data['Cumulative_Inflation']
        )
    else:
        stock_data['Inflation_Adjusted_Close'] = stock_data['Close']

    stock_data['Return_Adjusted'] = stock_data['Inflation_Adjusted_Close'].pct_change()
    stock_data['Volatility_Adjusted'] = stock_data['Return_Adjusted'].rolling(window=volatility_window).std() * (252 ** 0.5)
    vol_clean = stock_data['Volatility_Adjusted'].dropna()
    if vol_clean.empty:
        return None, None
    latest_volatility = vol_clean.iloc[-1]

    display_name = f'{ticker[:10]}...' if len(ticker) > 10 else ticker
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Inflation_Adjusted_Close'], mode='lines',
                              name=f'{display_name} Precio Ajustado', line=dict(color=colors[0], width=1.5), yaxis='y1',
                              hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Precio: %{y:.2f} ARS<extra></extra>'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Volatility_Adjusted'], mode='lines',
                              name=f'{display_name} Volatilidad', line=dict(color=colors[1 % len(colors)], width=1.5), yaxis='y2',
                              hovertemplate='Fecha: %{x|%Y-%m-%d}<br>Volatilidad: %{y:.2%}<extra></extra>'))

    for split in st.session_state.get("custom_splits", []):
        if split["ticker"] == ticker:
            fig.add_vline(x=datetime.combine(split["date"], datetime.min.time()).timestamp() * 1000,
                          line=dict(color="white", width=1, dash="dash"),
                          annotation_text=f"Split {split['ratio']}:1", annotation_position="top")

    fig.add_annotation(text="MTaurus - X: mtaurus_ok", xref="paper", yref="paper", x=0.02, y=0.02,
                        showarrow=False, font=dict(size=20, color="rgba(255,255,255,0.2)"), opacity=0.1)
    fig.update_layout(
        title=dict(text=f'Precio Ajustado por Inflación y Volatilidad Histórica de {display_name}', font=dict(size=20, color='white')),
        xaxis_title=dict(text='Fecha', font=dict(size=14, color='white')),
        yaxis=dict(title='Precio de Cierre Ajustado (ARS)', tickfont=dict(color='white'), tickformat=",.2f", ticksuffix=" ARS"),
        yaxis2=dict(title='Volatilidad Histórica (Anualizada)', tickfont=dict(color='white'), tickformat=".2%",
                    overlaying='y', side='right'),
        **PLOT_STYLE
    )
    return fig, latest_volatility
