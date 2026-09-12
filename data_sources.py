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
import requests
import urllib3
import yfinance as yf

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
