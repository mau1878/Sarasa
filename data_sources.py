"""
data_sources.py
================
Fetcher unificado con fallback en cascada para todos los módulos de MTaurus Chart Lab.

Orden de fallback (source='auto'): yfinance -> analisistecnico -> iol -> byma -> stooq
Cada función de fuente devuelve un DataFrame indexado por fecha con una única
columna (nombre del ticker, con '.' reemplazado por '_').

También centraliza:
- El histórico extendido de ^MERV (Stooq 1988-1996; requiere data/mrv_historico.txt)
- El CCL histórico de ^MERV (BCRA hasta 24/3/2003 + ratio YPFD.BA/YPF desde esa
  fecha; requiere data/merval_ccl_historico.csv)
- Un CCL "moderno" genérico (YPFD.BA/YPF o GD30/GD30C) para dolarizar cualquier ticker
- Ajuste de precios por splits (mismo diccionario que ya usaban tus otros scripts)
- Cálculo de variaciones por período (mensual/trimestral/anual), reutilizable
  desde heatmap, histogramas, rankings, etc.
"""

import os
import io
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests
import urllib3
import yfinance as yf

try:
    from num2words import num2words
    _HAS_NUM2WORDS = True
except ImportError:
    _HAS_NUM2WORDS = False

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Cache decorator opcional: funciona con o sin Streamlit en el proceso ---
try:
    import streamlit as st

    def cache_data(ttl=None):
        return st.cache_data(ttl=ttl) if ttl else st.cache_data
except ImportError:
    def cache_data(ttl=None):
        def deco(f):
            return f
        return deco

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
STOOQ_COOKIES_FILE = Path("stooq_cookies.json")
MERVAL_HISTORICO_PATH = os.path.join(DATA_DIR, "mrv_historico.txt")
MERVAL_HISTORICO_CUTOFF = pd.Timestamp("1996-10-08")  # primer día con datos de yfinance
MERVAL_CCL_HISTORICO_PATH = os.path.join(DATA_DIR, "merval_ccl_historico.csv")
MERVAL_CCL_HISTORICO_CUTOFF = pd.Timestamp("2003-03-24")  # desde acá manda YPFD.BA/YPF

CPI_AR_URL = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nargentina2.csv"
CPI_US_URL = "https://raw.githubusercontent.com/mau1878/Inflacion/refs/heads/main/inflaci%C3%B3nUSA.csv"
CPI_AR_FALLBACK_PATH = os.path.join(DATA_DIR, "cpi_argentina.csv")
CPI_US_FALLBACK_PATH = os.path.join(DATA_DIR, "cpi_usa.csv")

SPLITS = {
    'ADGO.BA': 1, 'ADBE.BA': 2, 'AEM.BA': 2, 'AMGN.BA': 3, 'AAPL.BA': 2, 'BAC.BA': 2,
    'GOLD.BA': 2, 'BIOX.BA': 2, 'CVX.BA': 2, 'LLY.BA': 7, 'XOM.BA': 2, 'FSLR.BA': 6,
    'IBM.BA': 3, 'JD.BA': 2, 'JPM.BA': 3, 'MELI.BA': 2, 'NFLX.BA': 3, 'PEP.BA': 3,
    'PFE.BA': 2, 'PG.BA': 3, 'RIO.BA': 2, 'SONY.BA': 2, 'SBUX.BA': 3, 'TXR.BA': 2,
    'BA.BA': 4, 'TM.BA': 3, 'VZ.BA': 2, 'VIST.BA': 3, 'WMT.BA': 3, 'AGRO.BA': (6, 2.1),
    'ECOG.BA': 10,
}


def _var_name(ticker):
    return ticker.replace(".", "_")


def ajustar_precios_por_splits(df, ticker, price_col):
    if df.empty or ticker not in SPLITS:
        return df
    df = df.copy()
    adj = SPLITS[ticker]
    ajustes = []
    if isinstance(adj, tuple):
        ajustes.append((datetime(2023, 11, 3), adj[0], "divide"))
        ajustes.append((datetime(2023, 11, 3), adj[1], "multiply"))
    else:
        fecha_split = datetime(2025, 8, 19) if ticker == "ECOG.BA" else datetime(2024, 1, 23)
        ajustes.append((fecha_split, adj, "divide"))
    for fecha, ratio, op in ajustes:
        mask = df.index < fecha
        if op == "divide":
            df.loc[mask, price_col] /= ratio
        else:
            df.loc[mask, price_col] *= ratio
    return df


# ============================= Fuentes individuales =============================

def _fetch_yfinance(ticker, start, end):
    try:
        session = cffi_requests.Session(impersonate="chrome124") if _HAS_CURL_CFFI else None
        df = yf.download(ticker, start=start, end=end, progress=False,
                          auto_adjust=False, session=session)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        col = "Close" if "Close" in df.columns else df.columns[0]
        var_name = _var_name(ticker)
        out = df[[col]].rename(columns={col: var_name})
        return ajustar_precios_por_splits(out, ticker, var_name)
    except Exception as e:
        logger.warning(f"yfinance falló para {ticker}: {e}")
        return pd.DataFrame()


def _fetch_analisistecnico(ticker, start, end):
    try:
        from_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.max.time()).timestamp())
        cookies = {'i18next': 'es'}
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        params = {'symbol': ticker.replace('.BA', ''), 'resolution': 'D',
                  'from': str(from_ts), 'to': str(to_ts)}
        r = requests.get('https://analisistecnico.com.ar/services/datafeed/history',
                          params=params, cookies=cookies, headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get('s') != 'ok':
            return pd.DataFrame()
        var_name = _var_name(ticker)
        df = pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), var_name: data['c']})
        df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
        return ajustar_precios_por_splits(df, ticker, var_name)
    except Exception as e:
        logger.warning(f"analisistecnico falló para {ticker}: {e}")
        return pd.DataFrame()


def _fetch_iol(ticker, start, end):
    try:
        from_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.max.time()).timestamp())
        cookies = {'isLogged': '1'}
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        params = {'symbolName': ticker.replace('.BA', ''), 'exchange': 'BCBA',
                  'from': str(from_ts), 'to': str(to_ts), 'resolution': 'D'}
        r = requests.get('https://iol.invertironline.com/api/cotizaciones/history',
                          params=params, cookies=cookies, headers=headers, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get('status') != 'ok' or 'bars' not in data:
            return pd.DataFrame()
        var_name = _var_name(ticker)
        df = pd.DataFrame(data['bars'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('Date').rename(columns={'close': var_name})[[var_name]]
        return ajustar_precios_por_splits(df, ticker, var_name)
    except Exception as e:
        logger.warning(f"IOL falló para {ticker}: {e}")
        return pd.DataFrame()


def _fetch_byma(ticker, start, end):
    try:
        from_ts = int(datetime.combine(start, datetime.min.time()).timestamp())
        to_ts = int(datetime.combine(end, datetime.max.time()).timestamp())
        symbol = ticker.replace('.BA', '') + ' 24HS'
        params = {'symbol': symbol, 'resolution': 'D', 'from': str(from_ts), 'to': str(to_ts)}
        headers = {'Accept': 'application/json', 'Referer': 'https://open.bymadata.com.ar/'}
        r = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params, headers=headers, verify=False, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get('s') != 'ok':
            return pd.DataFrame()
        var_name = _var_name(ticker)
        df = pd.DataFrame({'Date': pd.to_datetime(data['t'], unit='s'), var_name: data['c']})
        df = df.set_index('Date')
        return ajustar_precios_por_splits(df, ticker, var_name)
    except Exception as e:
        logger.warning(f"ByMA falló para {ticker}: {e}")
        return pd.DataFrame()


def _cargar_cookies_stooq():
    if not STOOQ_COOKIES_FILE.exists():
        return {}
    data = json.loads(STOOQ_COOKIES_FILE.read_text(encoding="utf-8"))
    return {c["name"]: c["value"] for c in data.get("cookies", []) if "stooq.com" in c.get("domain", "")}


def _fetch_stooq(ticker, start, end):
    cookies = _cargar_cookies_stooq()
    if not cookies:
        logger.warning("Sin cookies de Stooq cacheadas — correr refrescar_cookies_stooq.py primero")
        return pd.DataFrame()
    try:
        params = {'s': ticker.lower(), 'd1': start.strftime('%Y%m%d'),
                  'd2': end.strftime('%Y%m%d'), 'i': 'd'}
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        r = requests.get('https://stooq.com/q/d/l/', params=params, headers=headers,
                          cookies=cookies, timeout=15)
        if r.status_code != 200 or not r.text.startswith('Date'):
            return pd.DataFrame()
        df = pd.read_csv(io.StringIO(r.text))
        if df.empty or 'Close' not in df.columns:
            return pd.DataFrame()
        var_name = _var_name(ticker)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[['Date', 'Close']].rename(columns={'Close': var_name}).set_index('Date')
        return ajustar_precios_por_splits(df, ticker, var_name)
    except Exception as e:
        logger.warning(f"Stooq falló para {ticker}: {e}")
        return pd.DataFrame()


_SOURCES = {
    'yfinance': _fetch_yfinance,
    'analisistecnico': _fetch_analisistecnico,
    'iol': _fetch_iol,
    'byma': _fetch_byma,
    'stooq': _fetch_stooq,
}
_FALLBACK_ORDER = ['yfinance', 'analisistecnico', 'iol', 'byma', 'stooq']


@cache_data(ttl=86400)
def fetch_price_series(ticker, start, end, source='auto'):
    """
    Devuelve un DataFrame de 1 columna (nombre = ticker con '.' -> '_') con el
    precio de cierre de `ticker` entre `start` y `end` (ambos date/datetime).

    source='auto'  -> cascada yfinance -> analisistecnico -> iol -> byma -> stooq
    source=<nombre> -> usa solo esa fuente puntual
    """
    ticker = ticker.upper()
    df = pd.DataFrame()
    if source == 'auto':
        for src in _FALLBACK_ORDER:
            df = _SOURCES[src](ticker, start, end)
            if not df.empty:
                if src != 'yfinance':
                    logger.info(f"{ticker}: yfinance falló, se usó '{src}' como fallback")
                break
    else:
        df = _SOURCES.get(source, _fetch_yfinance)(ticker, start, end)

    df = extender_con_historico_merval(df, ticker, start)
    return df


# ===================== Histórico extendido ^MERV (Stooq 1988-1996) =====================

@cache_data()
def _cargar_historico_merval_stooq():
    try:
        hist = pd.read_csv(MERVAL_HISTORICO_PATH)
        hist['Date'] = pd.to_datetime(hist['<DATE>'], format='%Y%m%d')
        hist = hist[hist['Date'] < MERVAL_HISTORICO_CUTOFF]
        hist = hist[['Date', '<CLOSE>']].sort_values('Date').drop_duplicates('Date')
        return hist.set_index('Date')['<CLOSE>']
    except Exception as e:
        logger.warning(f"No se pudo cargar histórico Merval (Stooq): {e}")
        return pd.Series(dtype=float)


def extender_con_historico_merval(df, ticker, start_date):
    """Completa 1988-1996 en ^MERV con el histórico de Stooq (^MRV), si aplica."""
    if ticker.upper() != '^MERV':
        return df
    start_ts = pd.Timestamp(start_date)
    if start_ts >= MERVAL_HISTORICO_CUTOFF:
        return df
    hist = _cargar_historico_merval_stooq()
    if hist.empty:
        return df
    hist = hist[hist.index >= start_ts]
    if hist.empty:
        return df
    var_name = _var_name(ticker)
    hist_df = hist.to_frame(name=var_name)
    if df.empty:
        return hist_df
    return pd.concat([hist_df, df[~df.index.isin(hist_df.index)]]).sort_index()


# ===================== CCL (dolarización) =====================

@cache_data(ttl=86400)
def _descargar_ypfd_ypf_crudo():
    """YPFD.BA/YPF sin ajuste por dividendos: el ADR cobra dividendos en USD y
    la acción local en ARS, así que sus historiales de ajuste no son
    comparables y distorsionan el ratio si se usa Adj Close."""
    try:
        ypfd = yf.download('YPFD.BA', start='1996-01-01', progress=False, auto_adjust=False)
        ypf = yf.download('YPF', start='1996-01-01', progress=False, auto_adjust=False)

        def get_close(d):
            if isinstance(d.columns, pd.MultiIndex):
                d = d.droplevel(1, axis=1)
            return d['Close'] if 'Close' in d.columns else pd.Series(dtype=float)

        return get_close(ypfd).dropna(), get_close(ypf).dropna()
    except Exception as e:
        logger.warning(f"Error descargando YPFD.BA/YPF: {e}")
        return pd.Series(dtype=float), pd.Series(dtype=float)


def _ratio_ypfd_ypf(start, end):
    ypfd, ypf = _descargar_ypfd_ypf_crudo()
    if ypfd.empty or ypf.empty:
        return pd.Series(dtype=float)
    combined = pd.DataFrame({'YPFD': ypfd, 'YPF': ypf}).dropna()
    combined = combined[combined.index >= pd.Timestamp(start)]
    if combined.empty:
        return pd.Series(dtype=float)
    return (combined['YPFD'] * 10) / combined['YPF']  # 1 ADR YPF = 10 acciones locales YPFD.BA


@cache_data()
def _cargar_ccl_historico_merval():
    try:
        hist = pd.read_csv(MERVAL_CCL_HISTORICO_PATH, parse_dates=['Date'])
        hist = hist.sort_values('Date').drop_duplicates('Date').set_index('Date')
        return hist['TipoCambio']
    except Exception as e:
        logger.warning(f"No se pudo cargar CCL histórico Merval: {e}")
        return pd.Series(dtype=float)


def get_ccl_series(index, source='yfinance'):
    """
    Serie de CCL alineada al índice de fechas dado.
    - yfinance: para fechas < 24/3/2003 usa el tramo histórico BCRA; desde esa
      fecha usa el ratio YPFD.BA/YPF.
    - otras fuentes: usa GD30/GD30C (sin tramo histórico).
    """
    if len(index) == 0:
        return pd.Series(dtype=float)
    start, end = index.min(), index.max()

    if source != 'yfinance':
        gd30 = fetch_price_series('GD30', start, end, source=source)
        gd30c = fetch_price_series('GD30C', start, end, source=source)
        if gd30.empty or gd30c.empty:
            return pd.Series(dtype=float)
        ratio = gd30[_var_name('GD30')] / gd30c[_var_name('GD30C')]
        return ratio.reindex(index).ffill()

    ratio = pd.Series(index=index, dtype=float)
    if start < MERVAL_CCL_HISTORICO_CUTOFF:
        hist_ccl = _cargar_ccl_historico_merval()
        idx_hist = index[index < MERVAL_CCL_HISTORICO_CUTOFF]
        if not hist_ccl.empty:
            ratio.loc[idx_hist] = hist_ccl.reindex(idx_hist)
    idx_moderno = index[index >= MERVAL_CCL_HISTORICO_CUTOFF]
    if len(idx_moderno) > 0:
        ratio_ypf = _ratio_ypfd_ypf(idx_moderno.min(), idx_moderno.max())
        if not ratio_ypf.empty:
            ratio.loc[idx_moderno] = ratio_ypf.reindex(idx_moderno)
    return ratio.ffill()


def apply_ratio_and_ccl(main_df, main_ticker, second_ticker=None, third_ticker=None,
                         apply_ccl=False, source='yfinance', start=None, end=None):
    """
    Toma el DataFrame de 1 columna del ticker principal y, si corresponde, lo
    divide por un segundo/tercer ticker y/o por el CCL. Devuelve una Series.
    """
    var_main = _var_name(main_ticker)
    result = main_df[var_main].copy()

    if apply_ccl:
        ccl = get_ccl_series(result.index, source=source)
        if not ccl.empty:
            result = result / ccl.reindex(result.index).ffill()
        else:
            logger.warning("No se pudo obtener el CCL; se continúa sin dolarizar.")

    for extra_ticker in (second_ticker, third_ticker):
        if not extra_ticker:
            continue
        extra_df = fetch_price_series(
            extra_ticker, start or result.index.min(), end or result.index.max(), source=source
        )
        if extra_df.empty:
            logger.warning(f"No se pudo obtener {extra_ticker}; se ignora ese divisor.")
            continue
        extra_series = extra_df[_var_name(extra_ticker)].reindex(result.index).ffill()
        result = result / extra_series

    return result.dropna()


# ===================== Variaciones por período (reutilizable) =====================

def compute_period_changes(series, freq='ME'):
    """% de cambio por período (freq='ME' mensual, 'QE' trimestral), último vs último."""
    period_price = series.resample(freq).last()
    return period_price.pct_change() * 100


def compute_yearly_changes(series):
    """% de cambio anual (último vs último), indexado por año (int)."""
    yearly_price = series.resample('YE').last()
    yearly_change = yearly_price.pct_change() * 100
    yearly_change.index = yearly_change.index.year
    return yearly_change


# ===================== CPI / Inflación (con fallback local) =====================

def _parse_cpi_csv(raw):
    df = pd.read_csv(raw)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').set_index('Date')
    df['Cumulative_Inflation'] = (1 + df['CPI_MoM']).cumprod()
    daily = df['Cumulative_Inflation'].resample('D').ffill().interpolate(method='linear')
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    return daily


@cache_data(ttl=86400)
def cargar_moneda_historica_ar():
    """
    Serie diaria (forward-fill) con el nombre de la moneda de curso legal
    vigente en cada fecha en Argentina (columna 'Currency' del CSV de IPC AR:
    Peso Moneda Nacional / Peso Ley 18.188 / Peso Argentino / Austral / Peso).
    Con el mismo fallback remoto->local que cargar_cpi.
    """
    def _read(source):
        df = pd.read_csv(source)
        if 'Currency' not in df.columns:
            return pd.Series(dtype=str)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['Date']).sort_values('Date').set_index('Date')
        daily = df['Currency'].resample('D').ffill()
        daily.index = pd.to_datetime(daily.index).tz_localize(None)
        return daily

    try:
        return _read(CPI_AR_URL)
    except Exception as e:
        logger.warning(f"No se pudo bajar la serie de moneda histórica: {e}. Usando snapshot local.")
        try:
            return _read(CPI_AR_FALLBACK_PATH)
        except Exception as e2:
            logger.error(f"Tampoco se pudo leer el snapshot local de moneda: {e2}")
            return pd.Series(dtype=str)


def _extrapolar_hasta_hoy(cpi_mensual, meses_promedio=3):
    """
    Recibe un DataFrame mensual con columna 'CPI_MoM' indexado por fecha.
    Si el último dato real es de un mes anterior al actual, agrega filas
    mensuales sintéticas hasta hoy usando el promedio de los últimos
    `meses_promedio` meses reales. Devuelve (df_extendido, ultima_fecha_real).
    """
    cpi_mensual = cpi_mensual.sort_index()
    ultima_fecha_real = cpi_mensual.index.max()
    tasa_promedio = cpi_mensual['CPI_MoM'].tail(meses_promedio).mean()

    hoy = pd.Timestamp(datetime.now().date())
    if ultima_fecha_real >= hoy.to_period('M').to_timestamp():
        return cpi_mensual, ultima_fecha_real

    fechas_sinteticas = pd.date_range(
        start=ultima_fecha_real + pd.offsets.MonthBegin(1), end=hoy, freq='MS'
    )
    if len(fechas_sinteticas) == 0:
        return cpi_mensual, ultima_fecha_real

    filas_sinteticas = pd.DataFrame(
        {'CPI_MoM': [tasa_promedio] * len(fechas_sinteticas)}, index=fechas_sinteticas
    )
    cpi_extendido = pd.concat([cpi_mensual[['CPI_MoM']], filas_sinteticas])
    return cpi_extendido[~cpi_extendido.index.duplicated(keep='first')], ultima_fecha_real


def _construir_serie_diaria(cpi_mensual_extendido):
    cpi = cpi_mensual_extendido.sort_index().copy()
    cpi['Cumulative_Inflation'] = (1 + cpi['CPI_MoM']).cumprod()
    hoy = pd.Timestamp(datetime.now().date())
    if cpi.index.max() < hoy:
        cpi.loc[hoy] = np.nan
        cpi = cpi.sort_index()
    daily = cpi['Cumulative_Inflation'].resample('D').interpolate(method='linear').ffill()
    daily.index = pd.to_datetime(daily.index).tz_localize(None)
    return daily


@cache_data(ttl=86400)
def cargar_cpi(pais='AR'):
    """
    Serie diaria de inflación acumulada (índice, no %) para 'AR' o 'US'.

    Cascada de fuentes:
    - AR: API argentinadatos.com (INDEC) -> se completa 2007-2016 con el CSV
      curado (por la manipulación histórica del INDEC en ese tramo) -> si la
      API falla directamente, CSV de GitHub -> snapshot local.
    - US: FRED (serie CPIAUCSL, requiere FRED_API_KEY como variable de
      entorno) -> se completa pre-1947 con el CSV (la serie de FRED no
      empieza antes) -> si FRED falla, CSV de GitHub -> snapshot local.
    En ambos casos, si el último dato real es de un mes anterior al actual,
    se extrapola hasta hoy con el promedio de los últimos 3 meses reales.
    Devuelve también, en el índice de retorno, hasta qué fecha el dato es
    real (ver cargar_cpi_meta).
    """
    try:
        if pais == 'AR':
            cpi_mensual = _cpi_ar_desde_api()
        else:
            cpi_mensual = _cpi_us_desde_fred()
        cpi_extendido, ultima_fecha_real = _extrapolar_hasta_hoy(cpi_mensual)
        _CPI_META[pais] = ultima_fecha_real
        return _construir_serie_diaria(cpi_extendido)
    except Exception as e:
        logger.warning(f"Fuente en vivo de CPI ({pais}) falló: {e}. Usando CSV de GitHub/snapshot local.")
        try:
            daily = _parse_cpi_csv(CPI_AR_URL if pais == 'AR' else CPI_US_URL)
        except Exception as e2:
            logger.warning(f"CSV remoto de CPI ({pais}) también falló: {e2}. Usando snapshot local.")
            try:
                daily = _parse_cpi_csv(CPI_AR_FALLBACK_PATH if pais == 'AR' else CPI_US_FALLBACK_PATH)
            except Exception as e3:
                logger.error(f"Tampoco se pudo leer el snapshot local de CPI ({pais}): {e3}")
                return pd.Series(dtype=float)
        _CPI_META[pais] = daily.index.max()
        return daily


# Última fecha con dato REAL (no extrapolado) por país; se llena en cargar_cpi.
_CPI_META = {}


def cpi_ultima_fecha_real(pais='AR'):
    """Última fecha con dato oficial real (no extrapolado) cargada por cargar_cpi(pais)."""
    return _CPI_META.get(pais)


def _cpi_ar_desde_api():
    url = "https://api.argentinadatos.com/v1/finanzas/indices/inflacion"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    df = pd.DataFrame(r.json()).rename(columns={"fecha": "Date", "valor": "CPI_MoM_pct"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["CPI_MoM"] = df["CPI_MoM_pct"] / 100.0
    df = df.set_index("Date")[["CPI_MoM"]]

    # 2007-2016: se prefiere el CSV curado por la manipulación histórica del INDEC.
    try:
        csv = pd.read_csv(CPI_AR_URL)
        csv['Date'] = pd.to_datetime(csv['Date'], dayfirst=True, errors='coerce')
        csv = csv.dropna(subset=['Date']).set_index('Date')[['CPI_MoM']]
        mask_api = (df.index >= '2007-01-01') & (df.index <= '2016-12-31')
        df = df[~mask_api]
        mask_csv = (csv.index >= '2007-01-01') & (csv.index <= '2016-12-31')
        df = pd.concat([df, csv[mask_csv]]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
    except Exception as e:
        logger.warning(f"No se pudo aplicar el CSV curado 2007-2016 ({e}). Usando solo API.")
    return df


def _cpi_us_desde_fred():
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("FRED_API_KEY")
        except Exception:
            api_key = None
    if not api_key:
        raise ValueError("Falta FRED_API_KEY (variable de entorno o st.secrets).")

    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": "CPIAUCSL", "api_key": api_key, "file_type": "json"},
        timeout=15,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["observations"])[["date", "value"]]
    df = df.rename(columns={"date": "Date", "value": "CPI_Level"})
    df["Date"] = pd.to_datetime(df["Date"])
    df["CPI_Level"] = pd.to_numeric(df["CPI_Level"], errors="coerce")
    df = df.dropna(subset=["CPI_Level"]).set_index("Date").sort_index()
    df["CPI_MoM"] = df["CPI_Level"].pct_change()
    df = df.dropna(subset=["CPI_MoM"])[["CPI_MoM"]]

    # FRED (CPIAUCSL) arranca en 1947; se completa 1913-1946 con el CSV.
    try:
        csv = pd.read_csv(CPI_US_URL)
        csv['Date'] = pd.to_datetime(csv['Date'], dayfirst=True, errors='coerce')
        csv = csv.dropna(subset=['Date']).set_index('Date')[['CPI_MoM']]
        csv_pre = csv[csv.index < df.index.min()]
        df = pd.concat([csv_pre, df]).sort_index()
        df = df[~df.index.duplicated(keep='last')]
    except Exception as e:
        logger.warning(f"No se pudo completar el CPI de EE.UU. con el CSV pre-1947 ({e}). Usando solo FRED.")
    return df


# ===================== Redenominaciones de la moneda argentina =====================
# (fecha desde la que rige, ceros que se le sacaron a la moneda anterior, nombre)
REDENOMINATIONS = [
    (datetime(1970, 1, 1), 2, 'Peso Ley 18.188'),
    (datetime(1983, 6, 1), 4, 'Peso Argentino'),
    (datetime(1985, 6, 15), 3, 'Austral'),
    (datetime(1992, 1, 1), 4, 'Peso'),
]


def get_currency(fecha):
    """Nombre de la moneda de curso legal vigente en `fecha` (datetime)."""
    for change_date, _, currency in reversed(REDENOMINATIONS):
        if fecha >= change_date:
            return currency
    return 'Peso Moneda Nacional'


def to_current_peso(amount, fecha):
    """Convierte un monto en la moneda vigente en `fecha` a Pesos actuales
    (solo quita de ceros por redenominación, sin inflación)."""
    for change_date, zeroes, _ in REDENOMINATIONS:
        if fecha < change_date:
            amount /= 10 ** zeroes
    return amount


def from_current_peso(amount, fecha):
    """Convierte Pesos actuales a la moneda vigente en `fecha` (sin inflación)."""
    for change_date, zeroes, _ in reversed(REDENOMINATIONS):
        if fecha < change_date:
            amount *= 10 ** zeroes
    return amount


def format_arg_amount(amount, decimals=2):
    """Formatea un monto con separador de miles '.' y decimal ',' (estilo
    argentino). Si el valor es muy chico, agrega también notación científica."""
    if abs(amount) < 1e-6 and amount != 0:
        formatted_normal = f"{amount:,.12f}".replace(",", "X").replace(".", ",").replace("X", ".")
        formatted_scientific = f"{amount:.8e}".replace("e", "×10^")
        return formatted_normal, formatted_scientific
    formatted_normal = f"{amount:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return formatted_normal, None


def amount_to_words(amount, currency, decimals=2):
    """Escribe el monto en palabras (español), con centavos si corresponde.
    Requiere el paquete num2words; si no está instalado, devuelve un aviso."""
    if not _HAS_NUM2WORDS:
        return "(instalá el paquete 'num2words' para ver el monto en palabras)"
    if abs(amount) < 1e-6 and amount != 0:
        formatted_normal, _ = format_arg_amount(amount, 12)
        return f"Valor muy pequeño: {formatted_normal} {currency}"

    entero = int(round(amount))
    decimales = int(round((amount - entero) * (10 ** decimals)))

    try:
        word_part = num2words(entero, lang='es').capitalize()
    except OverflowError:
        try:
            word_part = num2words(entero, lang='en').capitalize() + " (en inglés)"
        except OverflowError:
            formatted_normal, _ = format_arg_amount(amount, decimals)
            return f"Valor demasiado grande para expresar en palabras: {formatted_normal} {currency}"

    if decimales > 0:
        try:
            decimal_words = num2words(decimales, lang='es').capitalize()
        except OverflowError:
            decimal_words = num2words(decimales, lang='en').capitalize() + " (en inglés)"
        return f"{word_part} {currency} con {decimal_words} centavos"
    return f"{word_part} {currency}"
