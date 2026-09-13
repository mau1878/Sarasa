import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta

from returns_evolution_plot import run_returns_evolution
from performance_plot import run_performance
from drawdown_plot import run_drawdown
from correlation_plot import run_correlation
from heatmap_script import run_heatmap
from inflation_adjusted_plot import (
    run_inflation_adjusted,
    compute_inflation_adjusted_df,
    build_plotly_figure,
    generate_matplotlib_plots,
)
from ratio_plot import run_ratio
from distribution_plot import (
    run_histogram, run_period_ranking, run_streaks,
    run_average_changes, run_yearly_ranking, run_descriptive_stats, run_period_line,
)
import data_sources
import stock_adjuster
import plotly.io as pio
import os
import json
import os


st.set_page_config(page_title="MTaurus Charts", layout="wide")

st.title("MTaurus – Chart Lab")

tool = st.sidebar.selectbox(
    "Módulo",
    [
        "Evolución de retornos",
        "Performance",
        "Drawdown",
        "Intradía",
        "Correlación",
        "Heatmap histórico",
        "Distribución (Histograma / Ranking / Rachas)",
        "Precio ajustado por inflación",
        "Calculadora de Inflación Histórica",
        "Ajustadora de Acciones por Inflación (Argentina)",
        "Ajustadora de Acciones por Inflación (EEUU)",
        "Cálculos Personalizados (Ratios)",
        "Análisis de Volatilidad",
        "Ratio entre activos",
    ],
)

FUENTES_DATOS = ["yfinance", "auto", "analisistecnico", "iol", "byma", "stooq"]

# ===================== Controles globales (paleta / CSV / splits / eventos) =====================
# Compartidos entre las 4 herramientas de "Ajustadora / Cálculos / Volatilidad",
# igual que en el script original (donde vivían en el sidebar, visibles siempre).
_HERRAMIENTAS_CON_CONTROLES_GLOBALES = {
    "Ajustadora de Acciones por Inflación (Argentina)",
    "Ajustadora de Acciones por Inflación (EEUU)",
    "Cálculos Personalizados (Ratios)",
    "Análisis de Volatilidad",
}

if tool in _HERRAMIENTAS_CON_CONTROLES_GLOBALES:
    stock_adjuster.init_session_state()

    st.sidebar.subheader("Paleta de colores de los gráficos")
    import seaborn as sns
    PALETAS_DISPONIBLES = {
        'Clásico': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'Deep (Seaborn)': sns.color_palette('deep', 10).as_hex(),
        'Muted (Seaborn)': sns.color_palette('muted', 10).as_hex(),
        'Bright (Seaborn)': sns.color_palette('bright', 10).as_hex(),
        'Pastel (Seaborn)': sns.color_palette('pastel', 10).as_hex(),
        'Colorblind (Seaborn)': sns.color_palette('colorblind', 10).as_hex(),
        'Dark (Seaborn)': sns.color_palette('dark', 10).as_hex(),
    }
    _paleta_sel = st.sidebar.selectbox(
        "Elegí la paleta de colores para las líneas de los gráficos:",
        list(PALETAS_DISPONIBLES.keys()), key='paleta_colores_select',
    )
    COLORS = PALETAS_DISPONIBLES[_paleta_sel]

    st.sidebar.subheader("Cargar datos personalizados desde CSV")
    _uploaded_files = st.sidebar.file_uploader(
        "Subir archivos CSV (deben tener columnas 'Date' y 'Close')",
        type=['csv'], accept_multiple_files=True, key='csv_uploader',
    )
    for _uf in _uploaded_files or []:
        _tk = _uf.name.lower().replace('_d.csv', '').replace('.csv', '').upper()
        try:
            _df = pd.read_csv(_uf, parse_dates=['Date'], index_col='Date')
            if 'Close' not in _df.columns:
                st.sidebar.error(f"El CSV {_uf.name} debe tener columna 'Close'.")
            else:
                st.session_state.uploaded_data[_tk] = _df[['Close']]
                st.sidebar.success(f"Datos cargados para {_tk} desde {_uf.name}")
        except Exception as _e:
            st.sidebar.error(f"Error al cargar {_uf.name}: {_e}")

    st.sidebar.subheader("Ajustes Manuales de Splits")
    _split_ticker = st.sidebar.text_input("Ticker para el split (ej: GLOB.BA):", key="custom_split_ticker_input")
    with st.sidebar.form(key="split_form"):
        _split_ratio = st.number_input("Ratio de split (ej: 3 para un split 3:1):", min_value=1.0, step=0.1, key="split_ratio_input")
        _split_date = st.date_input("Fecha del split:", min_value=datetime(2000, 1, 1).date(), max_value=datetime.now().date(), key="split_date_input")
        if st.form_submit_button("Agregar Split") and _split_ticker:
            st.session_state.custom_splits.append({"ticker": _split_ticker.strip().upper(), "ratio": _split_ratio, "date": _split_date})
            st.sidebar.success(f"Split agregado: {_split_ratio} en {_split_date} para {_split_ticker}")
    if st.session_state.custom_splits:
        st.sidebar.write("Splits Personalizados Agregados:")
        for _i, _split in enumerate(st.session_state.custom_splits):
            st.sidebar.write(f"Ticker: {_split['ticker']}, Ratio: {_split['ratio']}, Fecha: {_split['date']}")
            if st.sidebar.button(f"Eliminar Split {_i+1}", key=f"remove_split_{_i}"):
                st.session_state.custom_splits.pop(_i)
                st.rerun()

    st.sidebar.subheader("Eventos Personalizados")
    with st.sidebar.form(key="event_form"):
        _event_ticker = st.text_input("Ticker para el evento (ej: GLOB.BA):", key="event_ticker_input")
        _event_date = st.date_input("Fecha del evento:", min_value=datetime(2000, 1, 1).date(), max_value=datetime.now().date(), key="event_date_input")
        _event_desc = st.text_input("Descripción (ej: Ganancias Q4):", key="event_description_input")
        if st.form_submit_button("Agregar Evento") and _event_ticker and _event_desc:
            st.session_state.custom_events.append({"ticker": _event_ticker.strip().upper(), "date": _event_date, "description": _event_desc})
            st.sidebar.success(f"Evento agregado: {_event_desc} en {_event_date} para {_event_ticker}")
    if st.session_state.custom_events:
        st.sidebar.write("Eventos Personalizados Agregados:")
        for _i, _event in enumerate(st.session_state.custom_events):
            st.sidebar.write(f"Ticker: {_event['ticker']}, Evento: {_event['description']}, Fecha: {_event['date']}")
            if st.sidebar.button(f"Eliminar Evento {_i+1}", key=f"remove_event_{_i}"):
                st.session_state.custom_events.pop(_i)
                st.rerun()
else:
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

TICKERGROUPSFILE = "config/ticker_groups.json"
CORRELATIONFILE = "config/correlation_groups.json"
RATIOSFILE = "config/ratios.json"

def load_ticker_groups():
    if not os.path.exists(TICKERGROUPSFILE):
        return []
    with open(TICKERGROUPSFILE, "r", encoding="utf-8") as f:
        rawdata = json.load(f)
    processed = []
    if isinstance(rawdata, dict):
        for group_name, content in rawdata.items():
            tickers = []
            if isinstance(content, list):
                tickers = content
            elif isinstance(content, dict):
                tickers = list(content.keys())
            else:
                tickers = list(getattr(content, "keys", lambda: [])())
            tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
            if tickers:
                processed.append({"name": group_name, "tickers": list(dict.fromkeys(tickers))})
    return processed

def load_correlation_groups():
    """Load correlation groups: expected list of {'title': ..., 'tickers': [...] }."""
    if not os.path.exists(CORRELATIONFILE):
        return []
    with open(CORRELATIONFILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    groups = []
    if isinstance(data, list):
        for g in data:
            title = g.get("title")
            tickers = g.get("tickers") or []
            if title and tickers:
                groups.append(
                    {
                        "title": str(title),
                        "tickers": [t.strip().upper() for t in tickers if str(t).strip()],
                    }
                )
    return groups


def load_ratio_pairs():
    """Load predefined ratio pairs: list of {'main': ..., 'compare': ..., 'source': ...}."""
    if not os.path.exists(RATIOSFILE):
        return []
    with open(RATIOSFILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    pairs = []
    if isinstance(data, list):
        for r in data:
            main = r.get("main")
            compare = r.get("compare")
            source = (r.get("source") or "yfinance").lower()
            if main and compare:
                pairs.append(
                    {
                        "main": main.strip().upper(),
                        "compare": compare.strip().upper(),
                        "source": source,
                    }
                )
    return pairs




def fetch_data_yf_ratio(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    if "Adj Close" in data.columns:
        col = "Adj Close"
    elif "Close" in data.columns:
        col = "Close"
    else:
        return pd.DataFrame()
    return data[[col]].rename(columns={col: "Adj Close"})


def build_ratio_figure(main_ticker, compare_ticker, source="yfinance", start_year=None):
    end_date = datetime.now()
    if start_year:
        start_date = datetime.strptime(f"{start_year}-01-01", "%Y-%m-%d")
    else:
        start_date = end_date - timedelta(days=365 * 5)

    # Only yfinance in this version (no IOL) to keep logic simple
    df_main = fetch_data_yf_ratio(main_ticker, start_date, end_date)
    df_comp = fetch_data_yf_ratio(compare_ticker, start_date, end_date)

    if df_main.empty or df_comp.empty:
        return None

    common_index = df_main.index.intersection(df_comp.index)
    if len(common_index) < 50:
        return None

    main_series = df_main.loc[common_index, "Adj Close"]
    comp_series = df_comp.loc[common_index, "Adj Close"]

    ratio = main_series / comp_series

    ratio_mean = ratio.mean()
    ratio_std = ratio.std()
    rolling_mean = ratio.rolling(window=20).mean()
    rolling_std = ratio.rolling(window=20).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    upper_stat = ratio_mean + 2 * ratio_std
    lower_stat = ratio_mean - 2 * ratio_std

    fig = go.Figure()

    # Ratio
    fig.add_trace(
        go.Scatter(
            x=ratio.index,
            y=ratio,
            mode="lines",
            name=f"Ratio {main_ticker}/{compare_ticker}",
            line=dict(color="#40c0ff", width=2.2),
        )
    )

    # Rolling band (fill between)
    fig.add_trace(
        go.Scatter(
            x=upper_band.index,
            y=upper_band,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=lower_band.index,
            y=lower_band,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(64, 192, 255, 0.12)",
            line=dict(width=0),
            showlegend=False,
            name="Banda Bollinger",
        )
    )

    # Static stats
    fig.add_trace(
        go.Scatter(
            x=[ratio.index[0], ratio.index[-1]],
            y=[ratio_mean, ratio_mean],
            mode="lines",
            name="Media",
            line=dict(color="#a0a0ff", dash="dash", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ratio.index[0], ratio.index[-1]],
            y=[upper_stat, upper_stat],
            mode="lines",
            name="+2 Std",
            line=dict(color="#ff7070", dash="dot", width=1.4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[ratio.index[0], ratio.index[-1]],
            y=[lower_stat, lower_stat],
            mode="lines",
            name="-2 Std",
            line=dict(color="#60d070", dash="dot", width=1.4),
        )
    )

    # Prices on secondary axis
    for series, name, color in [
        (main_series, main_ticker, "#ffaa60"),
        (comp_series, compare_ticker, "#c080ff"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                mode="lines",
                name=name,
                opacity=0.45,
                yaxis="y2",
                line=dict(color=color, width=1.6),
            )
        )

    fig.update_layout(
        title=f"Ratio Analysis: {main_ticker} vs {compare_ticker}",
        yaxis=dict(
            title=dict(text="Ratio", font=dict(color="#d0d0ff")),
            showgrid=True,
            gridcolor="rgba(120,120,140,0.28)",
            zerolinecolor="rgba(180,180,200,0.18)",
            tickfont=dict(color="#d0d0ff"),
        ),
        yaxis2=dict(
            title=dict(text="Price (Log)", font=dict(color="#d0d0ff")),
            overlaying="y",
            side="right",
            type="log",
            showgrid=False,
            tickfont=dict(color="#d0d0ff"),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(35,35,55,0.65)",
            bordercolor="rgba(90,90,130,0.45)",
            borderwidth=1,
        ),
        template="plotly_dark",
        width=1600,
        height=800,
        plot_bgcolor="#0f0f1a",
        paper_bgcolor="#05050f",
        font=dict(color="#e0e0ff"),
    )

    fig.add_annotation(
        text="MTaurus - X: @mtaurus_ok",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=42, color="rgba(140,140,180,0.14)"),
        textangle=-30,
    )

    return fig

PRIMARY_COLOR = "#40c0ff"

plot_style = {
    "plot_bgcolor": "#0f0f1a",
    "paper_bgcolor": "#05050f",
    "font": dict(color="#e0e0ff", family="Arial", size=16),
    "template": "plotly_dark",
}


def build_inflation_adjusted_figure(ticker: str, start_year: int | None, source: str = "yfinance"):
    df = compute_inflation_adjusted_df(ticker, start_year, source)
    if df is None or df.empty:
        return None
    return build_plotly_figure(df, ticker)
from returns_evolution_plot import run_intraday_evolution
# ---- Evolución de retornos ----
# ---- Evolución de retornos ----
# ---- Evolución de retornos ----
# ---- Evolución de retornos ----
if tool == "Evolución de retornos":
    groups = load_ticker_groups()
    group_names = ["(ninguno)"] + [g["name"] for g in groups]
    selected_group = st.selectbox("Grupo predefinido (tickergroups.json)", group_names)
    
    tickers_str = st.text_input("Tickers manuales (coma)", "")
    period = st.selectbox("Período", ["YTD", "MTD", "WTD", "1Y", "2Y", "5Y"])
    
    # === NEW: Custom start date ===
    use_custom_start = st.checkbox("Usar fecha de inicio personalizada", value=False)
    custom_start_date = None
    if use_custom_start:
        custom_start_date = st.date_input(
            "Fecha de inicio (baseline para retornos)",
            value=datetime.now() - timedelta(days=365),
            min_value=datetime(1900, 1, 1).date(),
            max_value=datetime.now().date()
        )
    
    frequency_option = st.selectbox(
        "Frecuencia de retornos",
        ["Diaria", "Semanal", "Mensual"],
        index=0
    )
    freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}
    
    group_name = st.text_input("Nombre del grupo (si manual)", "Activos")
    custom_title = st.text_input("Título personalizado (opcional)", "")
    max_lines = st.slider("Máx. líneas", 5, 40, 20)
    
    col1, col2 = st.columns(2)
    with col1:
        smooth = st.checkbox("Suavizar curvas", True)
    with col2:
        adjust_ccl = st.checkbox("Ajustar por USD CCL (YPFD.BA/YPF)", False)
       
    exclude = st.text_input("Excluir tickers (coma)", "")
    
    if st.button("Generar gráfico"):
        tickers = []
        final_group_name = group_name
        
        if selected_group != "(ninguno)":
            g = next(g for g in groups if g["name"] == selected_group)
            tickers = g["tickers"]
            final_group_name = g["name"]
        if tickers_str.strip():
            manual_tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            tickers.extend(manual_tickers)
            if selected_group != "(ninguno)":
                final_group_name = f"{final_group_name} + manual"
        tickers = list(dict.fromkeys(tickers))
        
        if not tickers:
            st.warning("Elegí un grupo o ingresá al menos un ticker.")
        else:
            path = run_returns_evolution(
                tickers,
                period=period,
                group_name=final_group_name,
                custom_title=custom_title or None,
                max_lines=max_lines,
                exclude_tickers=exclude or None,
                smooth=smooth,
                ccl_adjust=adjust_ccl,
                frequency=freq_map[frequency_option],
                custom_start_date=custom_start_date  # ← New parameter
            )
            if path:
                st.image(path, width="stretch")
            else:
                st.warning("No se pudo generar el gráfico.")
elif tool == "Intradía":
    groups = load_ticker_groups()
    group_names = ["(ninguno)"] + [g["name"] for g in groups]
    selected_group = st.selectbox("Grupo predefinido (tickergroups.json)", group_names)
    
    tickers_str = st.text_input("Tickers manuales (coma)", "")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_date = st.date_input(
            "Fecha del gráfico intradía",
            value=datetime.now().date(),
            min_value=(datetime.now() - timedelta(days=30)).date(),
            max_value=datetime.now().date()
        )
    with col2:
        baseline = st.radio("Baseline", ["Apertura del día", "Cierre anterior"], horizontal=True)
    
    group_name = st.text_input("Nombre del grupo (si manual)", "Activos")
    custom_title = st.text_input("Título personalizado (opcional)", "")
    max_lines = st.slider("Máx. líneas", 5, 30, 15)
    smooth = st.checkbox("Suavizar curvas", True)
    
    if st.button("Generar gráfico intradía"):
        tickers = []
        final_group_name = group_name
        
        if selected_group != "(ninguno)":
            g = next(g for g in groups if g["name"] == selected_group)
            tickers = g["tickers"]
            final_group_name = g["name"]
        
        if tickers_str.strip():
            manual = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            tickers.extend(manual)
            if selected_group != "(ninguno)":
                final_group_name = f"{final_group_name} + manual"
        
        tickers = list(dict.fromkeys(tickers))
        
        if not tickers:
            st.warning("Elegí un grupo o ingresá al menos un ticker.")
        else:
            path = run_intraday_evolution(
                tickers,
                group_name=final_group_name,
                custom_title=custom_title or None,
                max_lines=max_lines,
                smooth=smooth,
                baseline="open" if baseline == "Apertura del día" else "close",
                target_date=selected_date   # ← NEW
            )
            if path and os.path.exists(path):
                st.image(path, width="stretch")
            else:
                st.warning("No se pudieron obtener datos para esa fecha (yfinance solo guarda ~7 días de 1m). Prueba con una fecha más reciente.")
# ---- Performance ----
# ---- Performance ----
elif tool == "Performance":
    groups = load_ticker_groups()
    group_names = ["(ninguno)"] + [g["name"] for g in groups]
    selected_group = st.selectbox("Grupo predefinido (tickergroups.json)", group_names)
    
    tickers_str = st.text_input("Tickers manuales (coma)", "")
    
    period = st.selectbox("Período", ["YTD", "1Y", "2Y", "5Y", "MTD", "WTD"])
    
    # === NEW: Custom start date ===
    use_custom_start = st.checkbox("Usar fecha de inicio personalizada", value=False)
    custom_start_date = None
    if use_custom_start:
        custom_start_date = st.date_input(
            "Fecha de inicio (baseline para performance)",
            value=datetime.now() - timedelta(days=365),
            min_value=datetime(1900, 1, 1).date(),
            max_value=datetime.now().date()
        )
    
    adjust_ccl = st.checkbox("Ajustar por USD CCL (YPFD.BA/YPF)", False)
    
    if st.button("Generar gráfico"):
        tickers = []
        group_name = "Activos"
        
        if selected_group != "(ninguno)":
            g = next(g for g in groups if g["name"] == selected_group)
            tickers = g["tickers"]
            group_name = g["name"]
        
        if tickers_str.strip():
            manual = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            tickers.extend(manual)
            if selected_group == "(ninguno)":
                group_name = "Custom"
            else:
                group_name = f"{group_name} + manual"
        
        if not tickers:
            st.warning("Elegí un grupo o ingresá al menos un ticker.")
        else:
            path = run_performance(
                tickers,
                period=period,
                group_name=group_name,
                tickers_file=TICKERGROUPSFILE,
                ccl_adjust=adjust_ccl,
                custom_start_date=custom_start_date  # ← NEW
            )
            if path and os.path.exists(path):
                st.image(path, width="stretch")
            else:
                st.warning("No se pudo generar el gráfico de performance.")


# ---- Drawdown ----
elif tool == "Drawdown":
    ticker = st.text_input("Ticker", "SPY")
    start_year = st.number_input("Año de inicio (opcional)", min_value=1900, max_value=2100, value=2000)
    from_last_ath = st.checkbox("Desde último ATH", False)

    if st.button("Generar gráfico"):
        path = run_drawdown(ticker, start_year=start_year, from_last_ath=from_last_ath)
        if path:
            st.image(path, width="stretch")
        else:
            st.warning("No se pudo generar el gráfico.")

# ---- Correlación ----
elif tool == "Correlación":
    groups = load_correlation_groups()
    group_titles = ["(ninguno)"] + [g["title"] for g in groups]

    selected_group = st.selectbox("Grupo predefinido (correlationgroups.json)", group_titles)
    tickers_str = st.text_input("Tickers manuales (coma)", "")
    custom_title = st.text_input("Título (opcional)", "")

    if st.button("Generar matriz"):
        tickers = []
        title = custom_title or "Acciones"

        if selected_group != "(ninguno)":
            g = next(g for g in groups if g["title"] == selected_group)
            tickers = g["tickers"]
            if not custom_title:
                title = g["title"]

        if tickers_str.strip():
            tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
            if not custom_title and selected_group == "(ninguno)":
                title = "Custom"

        if not tickers:
            st.warning("Elegí un grupo o ingresá al menos un ticker.")
        else:
            path = run_correlation(tickers, title=title)
            if path:
                st.image(path, width="stretch")
            else:
                st.warning("No se pudo generar la matriz.")


# ---- Heatmap histórico ----
elif tool == "Heatmap histórico":
    ticker = st.text_input("Ticker", "SPY")
    start_year = st.number_input("Año de inicio (opcional)", min_value=1900, max_value=2100, value=1990)
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)

    col1, col2 = st.columns(2)
    with col1:
        second_ticker = st.text_input("Segundo ticker (ratio, opcional)", "")
    with col2:
        third_ticker = st.text_input("Tercer ticker (ratio, opcional)", "")
    apply_ccl = st.checkbox(
        "Dolarizar por CCL (YPFD.BA/YPF con histórico BCRA para ^MERV, o GD30/GD30C en otras fuentes)",
        False,
    )

    if st.button("Generar heatmap"):
        path = run_heatmap(
            ticker,
            start_year=int(start_year) if start_year else None,
            source=source,
            second_ticker=second_ticker.strip().upper() or None,
            third_ticker=third_ticker.strip().upper() or None,
            apply_ccl=apply_ccl,
        )
        if path:
            st.image(path, width="stretch")
        else:
            st.warning("No se pudo generar el gráfico.")

# ---- Distribución (Histograma / Ranking / Rachas) ----
elif tool == "Distribución (Histograma / Ranking / Rachas)":
    ticker = st.text_input("Ticker", "GGAL.BA")
    start_year = st.number_input("Año de inicio (opcional)", min_value=1900, max_value=2100, value=2015)
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)
    analysis_period = st.radio("Período de análisis", ["Mes a Mes", "Trimestre a Trimestre"], horizontal=True)
    metric = st.radio("Métrica para el gráfico de cambios típicos", ["Promedio", "Mediana"], horizontal=True)

    col1, col2 = st.columns(2)
    with col1:
        second_ticker = st.text_input("Segundo ticker (ratio, opcional)", "")
    with col2:
        third_ticker = st.text_input("Tercer ticker (ratio, opcional)", "")
    apply_ccl = st.checkbox("Dolarizar por CCL", False)

    if st.button("Analizar distribución"):
        freq_label = "Mensual" if analysis_period == "Mes a Mes" else "Trimestral"
        kwargs = dict(
            start_year=int(start_year) if start_year else None,
            source=source,
            second_ticker=second_ticker.strip().upper() or None,
            third_ticker=third_ticker.strip().upper() or None,
            apply_ccl=apply_ccl,
        )

        # 1) Línea de tiempo de variaciones por período
        line_path = run_period_line(ticker, analysis_period=analysis_period, **kwargs)
        if line_path:
            st.image(line_path, width="stretch")

        # 2) Histograma + gaussiana
        hist_path = run_histogram(ticker, frequency=freq_label, **kwargs)
        if hist_path:
            st.image(hist_path, width="stretch")
        else:
            st.warning("No se pudo generar el histograma.")

        # 3) Cambios típicos (promedio o mediana) por mes/trimestre
        avg_path = run_average_changes(ticker, analysis_period=analysis_period, metric=metric, **kwargs)
        if avg_path:
            st.image(avg_path, width="stretch")

        # 4) Ranking mensual/trimestral (+ total de años, embebido)
        rank_path = run_period_ranking(ticker, analysis_period=analysis_period, **kwargs)
        if rank_path:
            st.image(rank_path, width="stretch")
        else:
            st.warning("No se pudo generar el ranking.")

        # 5) Ranking anual, separado (cantidad de meses/trimestres +/- por año)
        yearly_rank_path = run_yearly_ranking(ticker, analysis_period=analysis_period, **kwargs)
        if yearly_rank_path:
            st.image(yearly_rank_path, width="stretch")

        # 6) Estadísticas descriptivas
        stats = run_descriptive_stats(ticker, analysis_period=analysis_period, **kwargs)
        if stats:
            st.markdown("**Estadísticas descriptivas**")
            cols = st.columns(len(stats))
            for col, (label, value) in zip(cols, stats.items()):
                col.metric(label, value)

        # 7) Drawdown — reutiliza el módulo ya existente (mismo enfoque, sin duplicar)
        dd_path = run_drawdown(ticker, start_year=int(start_year) if start_year else None)
        if dd_path:
            st.image(dd_path, width="stretch")

        # 8) Rachas (tabular)
        with st.expander("📊 Análisis de rachas"):
            streaks = run_streaks(ticker, analysis_period=analysis_period, **kwargs)
            if streaks:
                st.dataframe(pd.DataFrame(streaks))
            else:
                st.info("Datos insuficientes para detectar rachas.")

# ---- Precio ajustado por inflación ----
elif tool == "Precio ajustado por inflación":
    ticker = st.text_input("Ticker", "AAPL.BA")
    start_year = st.number_input(
        "Año de inicio (opcional)",
        min_value=1900,
        max_value=2100,
        value=2000,
    )
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)

    if st.button("Generar gráfico"):
        fig = build_inflation_adjusted_figure(
            ticker.strip().upper(),
            int(start_year) if start_year else None,
            source=source,
        )
        if fig is None:
            st.warning("No se pudo generar el gráfico (datos insuficientes o CPI vacío).")
        else:
            st.plotly_chart(fig, use_container_width=True)

        path_lineal, path_log = generate_matplotlib_plots(
            ticker.strip().upper(),
            start_year=int(start_year) if start_year else None,
            source=source,
        )
        if path_lineal and path_log:
            st.markdown("**Versión Matplotlib (más liviana para celular):**")
            col1, col2 = st.columns(2)
            with col1:
                st.image(path_lineal, width="stretch")
            with col2:
                st.image(path_log, width="stretch")


# ---- Calculadora de Inflación Histórica ----
elif tool == "Calculadora de Inflación Histórica":
    st.subheader("Calculador de precios por inflación (Argentina)")

    st.markdown(
        """
Esta calculadora te dice cuánto valdría **hoy** una plata que tenías en el pasado
(o cuánto necesitabas en el pasado para comprar lo mismo que hoy).

**Dos efectos separados que hay que tener en cuenta:**
1. **Inflación**: con el tiempo, los precios suben y el dinero pierde poder de compra.
2. **Cambios de moneda**: además de la inflación, Argentina cambió de moneda varias
   veces y le "sacó ceros" a los billetes para simplificarlos:
   - Peso Moneda Nacional (hasta 1970)
   - Peso Ley 18.188 (1970 en adelante, se sacaron 2 ceros)
   - Peso Argentino (1983 en adelante, se sacaron 4 ceros)
   - Austral (1985 en adelante, se sacaron 3 ceros)
   - Peso (1992 en adelante, se sacaron 4 ceros — es la moneda actual)

   Por ejemplo: $10.000.000 de Pesos Moneda Nacional (antes de 1970) equivalen,
   **solo por los cambios de moneda** (sin contar la inflación), a $1 Peso actual.
        """
    )

    daily_cpi = data_sources.cargar_cpi("AR")
    ultima_fecha_real = data_sources.cpi_ultima_fecha_real("AR")
    hoy = pd.Timestamp(datetime.now().date())
    if ultima_fecha_real is not None and ultima_fecha_real < hoy.to_period("M").to_timestamp():
        st.caption(
            f"⚠️ IPC Argentina: último dato oficial publicado es de "
            f"{ultima_fecha_real.strftime('%B %Y')}. Los días posteriores usan una "
            f"estimación basada en el promedio de los últimos 3 meses."
        )

    if daily_cpi.empty:
        st.error("No se pudo cargar el IPC de Argentina (ni API en vivo, ni CSV remoto, ni snapshot local).")
    else:
        value_choice = st.radio(
            "¿Qué querés calcular?",
            ("Fecha de Inicio", "Fecha de Fin"),
            captions=[
                "Tengo un monto en el pasado y quiero saber cuánto vale hoy",
                "Tengo un monto de hoy y quiero saber cuánto necesitaba en el pasado",
            ],
            key="inflacion_calc_choice",
        )

        incluir_cambios_moneda = st.checkbox(
            "Tener en cuenta los cambios de moneda (además de la inflación)",
            value=True,
            help=(
                "Si lo dejás tildado, además del ajuste por inflación te muestro el "
                "equivalente en Pesos de hoy, aplicando también la quita de ceros de "
                "cada cambio de moneda. Si lo destildás, solo ves el ajuste por "
                "inflación, en la moneda de esa época."
            ),
            key="inflacion_calc_incluir_moneda",
        )

        st.caption(
            "⚠️ Estos cálculos son aproximados: usan el IPC mensual interpolado día a día. "
            "Cuanto más largo el período o más alta la inflación acumulada, menos exacto "
            "es el resultado en el día a día (aunque el número final es confiable)."
        )

        min_date = daily_cpi.index.min().date()
        max_date = daily_cpi.index.max().date()

        if value_choice == "Fecha de Inicio":
            start_date = st.date_input(
                "Selecciona la fecha de inicio:",
                min_value=min_date, max_value=max_date, value=min_date,
                key="inflacion_calc_start",
            )
            end_date = st.date_input(
                "Selecciona la fecha de fin:",
                min_value=min_date, max_value=max_date, value=max_date,
                key="inflacion_calc_end",
            )
            start_value = st.number_input(
                "Ingresa el monto que tenías en la fecha de inicio:",
                min_value=0.0, value=100.0, key="inflacion_calc_start_value",
            )

            start_dt = datetime.combine(start_date, datetime.min.time())
            moneda_inicio = data_sources.get_currency(start_dt)

            try:
                start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
                end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
                factor = end_inflation / start_inflation
                end_value_misma_moneda = start_value * factor

                start_fmt, _ = data_sources.format_arg_amount(start_value)
                end_fmt, end_fmt_sci = data_sources.format_arg_amount(end_value_misma_moneda)

                st.markdown("#### Resultado")
                st.write(f"**Monto original:** {moneda_inicio} {start_fmt} (al {start_date.strftime('%d/%m/%Y')})")
                st.caption(data_sources.amount_to_words(start_value, moneda_inicio))

                st.write(
                    f"**Ajustado solo por inflación**, en la misma moneda de esa época "
                    f"({moneda_inicio}): {moneda_inicio} {end_fmt}"
                    + (f" ({end_fmt_sci})" if end_fmt_sci else "")
                )
                st.caption(
                    f"Esto responde: ¿cuántos {moneda_inicio} necesitarías hoy, "
                    f"**en esa misma moneda vieja**, para tener el mismo poder de compra? "
                    "No tiene en cuenta que esa moneda ya no existe."
                )
                st.caption(data_sources.amount_to_words(end_value_misma_moneda, moneda_inicio))

                if incluir_cambios_moneda:
                    start_en_pesos_actuales = data_sources.to_current_peso(start_value, start_dt)
                    end_en_pesos_actuales = start_en_pesos_actuales * factor

                    pesos_ini_fmt, pesos_ini_sci = data_sources.format_arg_amount(start_en_pesos_actuales, 8)
                    pesos_fin_fmt, pesos_fin_sci = data_sources.format_arg_amount(end_en_pesos_actuales)

                    st.write("---")
                    st.write(
                        f"**Solo por el cambio de moneda** (sin inflación), esos "
                        f"{moneda_inicio} {start_fmt} equivalen hoy a: ARS {pesos_ini_fmt}"
                        + (f" ({pesos_ini_sci})" if pesos_ini_sci else "")
                    )
                    st.write(
                        f"**Resultado final (inflación + cambio de moneda), en Pesos actuales:** "
                        f"ARS {pesos_fin_fmt}" + (f" ({pesos_fin_sci})" if pesos_fin_sci else "")
                    )
                    st.caption(
                        "Este es el número más útil en la práctica: cuántos Pesos de hoy "
                        "necesitarías para tener el mismo poder de compra que tenías en la "
                        "fecha de inicio, contando todo (inflación y cambios de moneda)."
                    )
                    st.caption(data_sources.amount_to_words(end_en_pesos_actuales, "pesos"))
            except KeyError as e:
                st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")

        else:
            start_date = st.date_input(
                "Selecciona la fecha de inicio:",
                min_value=min_date, max_value=max_date, value=min_date,
                key="inflacion_calc_start2",
            )
            end_date = st.date_input(
                "Selecciona la fecha de fin:",
                min_value=start_date, max_value=max_date, value=max_date,
                key="inflacion_calc_end2",
            )
            end_value = st.number_input(
                "Ingresa el monto que tenés en la fecha de fin:",
                min_value=0.0, value=100.0, key="inflacion_calc_end_value",
            )

            start_dt = datetime.combine(start_date, datetime.min.time())
            end_dt = datetime.combine(end_date, datetime.min.time())
            moneda_inicio = data_sources.get_currency(start_dt)
            moneda_fin = data_sources.get_currency(end_dt)

            try:
                start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
                end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
                factor = end_inflation / start_inflation
                start_value_misma_moneda = end_value / factor

                end_fmt, _ = data_sources.format_arg_amount(end_value)
                start_fmt, start_fmt_sci = data_sources.format_arg_amount(start_value_misma_moneda)

                st.markdown("#### Resultado")
                st.write(f"**Monto de referencia:** {moneda_fin} {end_fmt} (al {end_date.strftime('%d/%m/%Y')})")
                st.caption(data_sources.amount_to_words(end_value, moneda_fin))

                st.write(
                    f"**Deflactado solo por inflación**, en la misma moneda ({moneda_fin}): "
                    f"{moneda_fin} {start_fmt}" + (f" ({start_fmt_sci})" if start_fmt_sci else "")
                )
                st.caption(
                    f"Esto responde: ¿cuántos {moneda_fin} necesitabas en la fecha de "
                    "inicio para tener el mismo poder de compra? Sin tener en cuenta "
                    "que en esa época podía existir otra moneda."
                )
                st.caption(data_sources.amount_to_words(start_value_misma_moneda, moneda_fin))

                if incluir_cambios_moneda:
                    end_en_pesos_actuales = data_sources.to_current_peso(end_value, end_dt)
                    start_en_pesos_actuales = end_en_pesos_actuales / factor
                    start_moneda_historica = data_sources.from_current_peso(start_en_pesos_actuales, start_dt)

                    pesos_fin_fmt, pesos_fin_sci = data_sources.format_arg_amount(end_en_pesos_actuales)
                    hist_fmt, hist_sci = data_sources.format_arg_amount(start_moneda_historica, 8)

                    st.write("---")
                    st.write(
                        f"**Solo por el cambio de moneda**, ese monto equivale hoy a: "
                        f"ARS {pesos_fin_fmt}" + (f" ({pesos_fin_sci})" if pesos_fin_sci else "")
                    )
                    st.caption(data_sources.amount_to_words(end_en_pesos_actuales, "pesos"))

                    st.write(
                        f"**Resultado final, en la moneda que circulaba el "
                        f"{start_date.strftime('%d/%m/%Y')} ({moneda_inicio}):** "
                        f"{moneda_inicio} {hist_fmt}" + (f" ({hist_sci})" if hist_sci else "")
                    )
                    st.caption(
                        f"Esto te dice cuántos billetes de {moneda_inicio} necesitabas en "
                        "esa fecha para comprar lo mismo que hoy — contando inflación y "
                        "cambios de moneda."
                    )
                    st.caption(data_sources.amount_to_words(start_moneda_historica, moneda_inicio))
            except KeyError as e:
                st.error(f"Error al obtener la inflación para las fechas seleccionadas: {e}")


# ---- Ajustadora de Acciones por Inflación (Argentina) ----
elif tool == "Ajustadora de Acciones por Inflación (Argentina)":
    st.subheader("Ajustadora de acciones por inflación (Argentina)")
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)

    tickers_input = st.text_input(
        "Ingresa los tickers separados por comas (ej: AAPL.BA, MSFT.BA, META):",
        key="tickers_input_arg",
    )
    sma_period = st.number_input("Períodos para el SMA del primer ticker:", min_value=1, value=10, key="sma_period_input_arg")

    daily_cpi_ar = data_sources.cargar_cpi("AR")
    if daily_cpi_ar.empty:
        st.error("No se pudo cargar el IPC de Argentina.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            plot_start_date = st.date_input(
                "Fecha de inicio del gráfico:",
                min_value=daily_cpi_ar.index.min().date(), max_value=daily_cpi_ar.index.max().date(),
                value=(daily_cpi_ar.index.max() - timedelta(days=365)).date(), key="plot_start_date_input_arg",
            )
        with col2:
            plot_end_date = st.date_input(
                "Fecha de fin del gráfico:",
                min_value=plot_start_date, max_value=daily_cpi_ar.index.max().date(),
                value=daily_cpi_ar.index.max().date(), key="plot_end_date_input_arg",
            )

        force_inflation_arg = st.checkbox("Aplicar ajuste por inflación a todos los tickers (incluyendo no-.BA)", False, key="force_inflation_arg")
        show_percentage = st.checkbox("Mostrar valores ajustados como porcentajes", False, key="show_percentage_arg")
        show_percentage_from_recent = st.checkbox("Porcentajes desde el valor más reciente", False, key="show_percentage_from_recent_arg")
        is_percentage_mode = show_percentage or show_percentage_from_recent
        use_log_scale_arg = st.checkbox("Escala logarítmica en el eje Y", False, key="use_log_scale_arg", disabled=is_percentage_mode) and not is_percentage_mode
        show_nominal_ghost_arg = st.checkbox("Incluir línea fantasma con el valor nominal", False, key="show_nominal_ghost_arg")

        if tickers_input:
            fig, fig_mpl, nominal_dict, adjusted_dict, ticker_var_map, errors = stock_adjuster.graficar_activos_ajustados(
                tickers_input=tickers_input, sma_period=sma_period, plot_start_date=plot_start_date,
                daily_cpi_serie=daily_cpi_ar, source=source, moneda="ARS", colors=COLORS,
                is_percentage_mode=is_percentage_mode, show_percentage_from_recent=show_percentage_from_recent,
                use_log_scale=use_log_scale_arg, show_nominal_ghost=show_nominal_ghost_arg,
                siempre_ajustar=False, force_inflation=force_inflation_arg, plot_end_date=plot_end_date,
            )
            for err in errors:
                st.error(err)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Versión Matplotlib (más liviana para celular):**")
            st.pyplot(fig_mpl)
            # Guardado para que "Cálculos Personalizados" pueda usar estos mismos datos
            st.session_state['sa_nominal_ar'] = nominal_dict
            st.session_state['sa_var_map_ar'] = ticker_var_map

# ---- Ajustadora de Acciones por Inflación (EEUU) ----
elif tool == "Ajustadora de Acciones por Inflación (EEUU)":
    st.subheader("Ajustadora de acciones por inflación (EEUU)")
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)

    tickers_input_us = st.text_input(
        "Ingresa los tickers separados por comas (ej: AAPL, MSFT, META.BA):",
        key="tickers_input_us",
    )
    sma_period_us = st.number_input("Períodos para el SMA del primer ticker:", min_value=1, value=10, key="sma_period_input_us")

    daily_us_cpi = data_sources.cargar_cpi("US")
    if daily_us_cpi.empty:
        st.error("No se pudo cargar el CPI de EE.UU.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            plot_start_date_us = st.date_input(
                "Fecha de inicio del gráfico:",
                min_value=daily_us_cpi.index.min().date(), max_value=daily_us_cpi.index.max().date(),
                value=(daily_us_cpi.index.max() - timedelta(days=365)).date(), key="plot_start_date_input_us",
            )
        with col2:
            plot_end_date_us = st.date_input(
                "Fecha de fin del gráfico:",
                min_value=plot_start_date_us, max_value=daily_us_cpi.index.max().date(),
                value=daily_us_cpi.index.max().date(), key="plot_end_date_input_us",
            )

        show_percentage_us = st.checkbox("Mostrar valores ajustados como porcentajes", False, key="show_percentage_us")
        show_percentage_from_recent_us = st.checkbox("Porcentajes desde el valor más reciente", False, key="show_percentage_from_recent_us")
        is_percentage_mode_us = show_percentage_us or show_percentage_from_recent_us
        use_log_scale_us = st.checkbox("Escala logarítmica en el eje Y", False, key="use_log_scale_us", disabled=is_percentage_mode_us) and not is_percentage_mode_us
        show_nominal_ghost_us = st.checkbox("Incluir línea fantasma con el valor nominal", False, key="show_nominal_ghost_us")

        if tickers_input_us:
            fig, fig_mpl, nominal_dict, adjusted_dict, ticker_var_map, errors = stock_adjuster.graficar_activos_ajustados(
                tickers_input=tickers_input_us, sma_period=sma_period_us, plot_start_date=plot_start_date_us,
                daily_cpi_serie=daily_us_cpi, source=source, moneda="USD", colors=COLORS,
                is_percentage_mode=is_percentage_mode_us, show_percentage_from_recent=show_percentage_from_recent_us,
                use_log_scale=use_log_scale_us, show_nominal_ghost=show_nominal_ghost_us,
                siempre_ajustar=True, plot_end_date=plot_end_date_us,
            )
            for err in errors:
                st.error(err)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Versión Matplotlib (más liviana para celular):**")
            st.pyplot(fig_mpl)

# ---- Cálculos Personalizados (Ratios) ----
elif tool == "Cálculos Personalizados (Ratios)":
    st.subheader("Cálculos o Ratios Personalizados")
    st.markdown(
        """
Definí expresiones matemáticas sobre los tickers que ya cargaste en
**"Ajustadora de Acciones por Inflación (Argentina)"** (corré esa herramienta
primero, con los tickers que quieras usar acá).

**Ejemplo:** `META*(YPFD.BA / YPF)/20`

- Usá los tickers tal como los ingresaste ahí (incluyendo `.BA` si corresponde).
- Los tickers con puntos (`.`) se reemplazan automáticamente por guiones bajos
  (`_`) al evaluar la expresión.
- Podés usar operadores matemáticos básicos: `+`, `-`, `*`, `/`, `**`, etc.
        """
    )

    nominal_dict = st.session_state.get('sa_nominal_ar')
    ticker_var_map = st.session_state.get('sa_var_map_ar')

    if not nominal_dict or not ticker_var_map:
        st.warning("Primero generá un gráfico en 'Ajustadora de Acciones por Inflación (Argentina)' con los tickers que quieras combinar.")
    else:
        custom_expression = st.text_input(
            "Expresión personalizada:", placeholder="Por ejemplo: META*(YPFD.BA / YPF)/20", key="custom_expression_input",
        )
        custom_title = st.text_input("Título personalizado del gráfico (opcional):", key="custom_title_input")
        show_percentage = st.checkbox("Mostrar como porcentaje", False, key="custom_show_percentage")
        show_percentage_from_recent = st.checkbox("Porcentaje desde el valor más reciente", False, key="custom_show_percentage_recent")
        use_log_scale = st.checkbox("Escala logarítmica", False, key="custom_use_log_scale")

        if custom_expression:
            try:
                daily_cpi_ar = data_sources.cargar_cpi("AR")
                fig = stock_adjuster.evaluate_custom_expression(
                    custom_expression, nominal_dict, ticker_var_map, daily_cpi_ar, COLORS,
                    custom_title=custom_title, show_percentage=show_percentage,
                    show_percentage_from_recent=show_percentage_from_recent, use_log_scale=use_log_scale,
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                available_vars = ', '.join(ticker_var_map.values())
                st.error(f"Error al evaluar la expresión: {e}\n\nVariables disponibles: {available_vars}")

# ---- Análisis de Volatilidad ----
elif tool == "Análisis de Volatilidad":
    st.subheader("Comparación de Volatilidad Histórica Ajustada por Inflación y Precio Ajustado (Argentina)")
    source = st.selectbox("Fuente de datos", FUENTES_DATOS, index=0)
    selected_ticker = st.text_input("Ticker a analizar (cualquiera, ej: AAPL, AAPL.BA):", key="selected_ticker_input")

    daily_cpi_ar = data_sources.cargar_cpi("AR")
    if daily_cpi_ar.empty:
        st.error("No se pudo cargar el IPC de Argentina.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            vol_start = st.date_input(
                "Fecha de inicio:", min_value=daily_cpi_ar.index.min().date(), max_value=daily_cpi_ar.index.max().date(),
                value=(daily_cpi_ar.index.max() - timedelta(days=365)).date(), key="vol_comparison_start_date_input",
            )
        with col2:
            vol_end = st.date_input(
                "Fecha de fin:", min_value=vol_start, max_value=daily_cpi_ar.index.max().date(),
                value=daily_cpi_ar.index.max().date(), key="vol_comparison_end_date_input",
            )
        volatility_window = st.number_input("Ventana para la volatilidad histórica (en períodos):", min_value=1, value=20, key="volatility_window_input")

        if selected_ticker:
            fig, latest_vol = stock_adjuster.build_volatility_figure(
                selected_ticker, vol_start, vol_end, source, daily_cpi_ar, COLORS, volatility_window=int(volatility_window),
            )
            if fig is None:
                st.error(f"No se encontraron datos suficientes para {selected_ticker}.")
            else:
                st.write(f"**Volatilidad Histórica Ajustada por Inflación (ventana {volatility_window}):** {latest_vol:.2%}")
                st.plotly_chart(fig, use_container_width=True)

# ---- Ratio entre activos ----
elif tool == "Ratio entre activos":
    pairs = load_ratio_pairs()
    preset_labels = ["(ninguno)"] + [f"{p['main']} / {p['compare']} ({p['source']})" for p in pairs]

    selected_pair_label = st.selectbox("Ratio predefinido (ratios.json)", preset_labels)
    main_ticker = st.text_input("Ticker principal (manual)", "")
    compare_ticker = st.text_input("Ticker comparado (manual)", "")
    start_year = st.number_input("Año de inicio (opcional)", min_value=1900, max_value=2100, value=2015)

    if st.button("Generar ratio"):
        main = None
        compare = None
        source = "yfinance"

        if selected_pair_label != "(ninguno)":
            idx = preset_labels.index(selected_pair_label) - 1
            p = pairs[idx]
            main, compare, source = p["main"], p["compare"], p["source"]

        if main_ticker.strip() and compare_ticker.strip():
            main = main_ticker.strip().upper()
            compare = compare_ticker.strip().upper()
            # keep source from preset or default yfinance

        if not main or not compare:
            st.warning("Elegí un par predefinido o ingresá ambos tickers manualmente.")
        else:
            fig = build_ratio_figure(
                main,
                compare,
                source=source,
                start_year=int(start_year) if start_year else None,
            )
            if fig is None:
                st.warning("No se pudo generar el gráfico de ratio (datos insuficientes).")
            else:
                st.plotly_chart(fig, use_container_width=True)


