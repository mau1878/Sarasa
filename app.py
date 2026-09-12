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
from distribution_plot import run_histogram, run_period_ranking, run_streaks
import data_sources
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
        "Ratio entre activos",
    ],
)

FUENTES_DATOS = ["yfinance", "auto", "analisistecnico", "iol", "byma", "stooq"]

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

        hist_path = run_histogram(ticker, frequency=freq_label, **kwargs)
        if hist_path:
            st.image(hist_path, width="stretch")
        else:
            st.warning("No se pudo generar el histograma.")

        rank_path = run_period_ranking(ticker, analysis_period=analysis_period, **kwargs)
        if rank_path:
            st.image(rank_path, width="stretch")
        else:
            st.warning("No se pudo generar el ranking.")

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
    st.subheader("Calculadora de precios por inflación (Argentina)")
    st.markdown(
        """
Convertí el poder adquisitivo de un monto en pesos argentinos entre dos fechas,
usando el IPC (INDEC) encadenado desde 1943 hasta hoy (con proyección hasta 2027).

**¿Por qué funciona esto a pesar de los cambios de moneda?** Entre 1943 y hoy la
Argentina cambió de signo monetario varias veces — *Peso Moneda Nacional* →
*Peso Ley 18.188* → *Peso Argentino* → *Austral* → *Peso* (convertible) — con
quitas de ceros en el medio (1970, 1983, 1985, 1992). El cálculo no usa el
nombre de la moneda ni sus quitas de ceros: encadena la variación **porcentual**
mes a mes del IPC, así que el resultado ya viene expresado en el poder
adquisitivo equivalente de hoy, sin que vos tengas que aplicar ningún factor de
conversión extra por la redenominación.
        """
    )

    daily_cpi = data_sources.cargar_cpi("AR")
    daily_moneda = data_sources.cargar_moneda_historica_ar()

    if daily_cpi.empty:
        st.error("No se pudo cargar el IPC de Argentina (ni remoto ni snapshot local).")
    else:
        min_date = daily_cpi.index.min().date()
        max_date = daily_cpi.index.max().date()

        def _moneda_de(fecha):
            if daily_moneda.empty:
                return None
            ts = pd.to_datetime(fecha)
            return daily_moneda.get(ts, None)

        value_choice = st.radio(
            "¿Querés ingresar el valor para la fecha de inicio o la fecha de fin?",
            ("Fecha de Inicio", "Fecha de Fin"),
            key="inflacion_calc_choice",
        )

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
                "Ingresa el valor en la fecha de inicio (en ARS):",
                min_value=0.0, value=100.0, key="inflacion_calc_start_value",
            )

            try:
                start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
                end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
                end_value = start_value * (end_inflation / start_inflation)

                moneda_inicio = _moneda_de(start_date)
                moneda_fin = _moneda_de(end_date)

                st.write(
                    f"Valor inicial el {start_date}: ARS {start_value:,.2f}"
                    + (f" ({moneda_inicio})" if moneda_inicio else "")
                )
                st.write(
                    f"Valor ajustado el {end_date}: ARS {end_value:,.2f}"
                    + (f" ({moneda_fin})" if moneda_fin else "")
                )
                if moneda_inicio and moneda_fin and moneda_inicio != moneda_fin:
                    st.info(
                        f"Entre esas dos fechas la moneda de curso legal pasó de "
                        f"'{moneda_inicio}' a '{moneda_fin}'. El resultado ya está "
                        f"expresado en pesos de hoy — no hace falta ningún ajuste extra."
                    )
            except KeyError as e:
                st.error(f"No hay datos de inflación para alguna de las fechas seleccionadas: {e}")
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
                "Ingresa el valor en la fecha de fin (en ARS):",
                min_value=0.0, value=100.0, key="inflacion_calc_end_value",
            )

            try:
                start_inflation = daily_cpi.loc[pd.to_datetime(start_date)]
                end_inflation = daily_cpi.loc[pd.to_datetime(end_date)]
                start_value = end_value / (end_inflation / start_inflation)

                moneda_inicio = _moneda_de(start_date)
                moneda_fin = _moneda_de(end_date)

                st.write(
                    f"Valor ajustado el {start_date}: ARS {start_value:,.2f}"
                    + (f" ({moneda_inicio})" if moneda_inicio else "")
                )
                st.write(
                    f"Valor final el {end_date}: ARS {end_value:,.2f}"
                    + (f" ({moneda_fin})" if moneda_fin else "")
                )
                if moneda_inicio and moneda_fin and moneda_inicio != moneda_fin:
                    st.info(
                        f"Entre esas dos fechas la moneda de curso legal pasó de "
                        f"'{moneda_inicio}' a '{moneda_fin}'. El resultado ya está "
                        f"expresado en pesos de hoy — no hace falta ningún ajuste extra."
                    )
            except KeyError as e:
                st.error(f"No hay datos de inflación para alguna de las fechas seleccionadas: {e}")


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


