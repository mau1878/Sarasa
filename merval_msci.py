"""
merval_msci.py
==============
Empalme del S&P Merval (hasta 16/10/2026) con el MSCI Merval (desde 19/10/2026).

Problema: desde el 19/10/2026 el índice pasó a ser el MSCI Merval, con base
100.000 = cierre del S&P Merval del 16/10/2026. yfinance (^MERV) no trae el
histórico diario del índice nuevo, solo el dato del día.

Solución: para ^MERV se conserva lo que traiga la fuente hasta el 16/10/2026
(escala S&P, ~3 millones) y desde el 17/10 se agrega la serie de Rava
(ya reescalada a base MSCI) convertida a escala S&P con un factor k:

    k = cierre S&P (yfinance) / cierre MSCI (Rava)     [medido en días solapados <= 16/10]

Así todo el histórico sigue en la MISMA escala de antes (CCL, ajuste por
inflación, Stooq 1988-96 y demás no cambian). Si algún día se prefiere la
escala MSCI, dividir toda la serie por k.

Uso (df con índice de fechas, o con columna 'Date', y una columna de precio):
    from merval_msci import empalmar_merval
    df = empalmar_merval(df)
"""
import logging
import os
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ANCHOR = pd.Timestamp("2026-10-16")  # último cierre con el índice S&P; MSCI = 100.000
RAVA_URL = "https://mercado.rava.com/api/prices/historico/ind/MERVAL"
_HEADERS = {
    "accept": "*/*",
    "accept-language": "es-AR,es;q=0.9",
    "referer": "https://mercado.rava.com/grafico?especie=MERVAL&mercado=ind&modo=historico",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
}
_BASE = os.path.dirname(os.path.abspath(__file__))
SEED_PATH = os.path.join(_BASE, "data", "merval_msci_rava.csv")  # Date,Close (respaldo + caché)
_TTL = 3600  # segundos: no pegarle a Rava más de una vez por hora por proceso
_mem = {"t": 0.0, "s": None}

# True  -> la serie ^MERV completa (incluido Stooq 1988-96) se expresa en escala MSCI
#          (16/10/2026 = 100.000), igual que se publica hoy el índice.
# False -> se mantiene la escala S&P histórica (~3 millones).
# Los retornos % son idénticos en ambos casos; solo cambia el nivel absoluto.
ESCALA_MSCI = True
_K = {"v": None}  # último factor S&P/MSCI calculado (lo usa a_escala_msci)


# ------------------------------------------------------------------ Rava
def _fetch_rava(dias=45):
    r = requests.get(RAVA_URL, params={"dias": dias}, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json()["datos"])
    idx = pd.to_datetime(df["fecha"], utc=True).dt.tz_localize(None).dt.normalize()
    return pd.Series(df["precio"].astype(float).values, index=idx).sort_index()


def _read_seed():
    try:
        s = pd.read_csv(SEED_PATH, parse_dates=["Date"]).set_index("Date")["Close"]
        return s.astype(float).sort_index()
    except Exception:
        return pd.Series(dtype=float)


def get_msci_merval(force=False):
    """Serie MSCI Merval (escala MSCI) = respaldo local + Rava (lo de Rava pisa)."""
    if not force and _mem["s"] is not None and time.time() - _mem["t"] < _TTL:
        return _mem["s"]
    seed = _read_seed()
    try:
        live = _fetch_rava()
        s = pd.concat([seed[~seed.index.isin(live.index)], live]).sort_index()
        try:  # persistir (en Streamlit Cloud el disco es efímero; no pasa nada si falla)
            os.makedirs(os.path.dirname(SEED_PATH), exist_ok=True)
            s.rename("Close").rename_axis("Date").to_csv(SEED_PATH)
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Rava no disponible ({e}); uso respaldo local")
        s = seed
    _mem["s"], _mem["t"] = s, time.time()
    return s


# ------------------------------------------------------------------ factor
def _factor(old_close, msci):
    """k = escala S&P / escala MSCI, en los días <= ANCHOR donde hay ambos datos."""
    ratio = (old_close.loc[:ANCHOR] / msci.loc[:ANCHOR]).dropna()
    if ratio.empty:
        return None
    cv = ratio.std() / ratio.mean() if len(ratio) > 1 else 0.0
    if cv > 0.01:
        logger.warning(f"Factor S&P/MSCI no constante (cv={cv:.2%}); revisar reescalado de Rava")
    return float(ratio.median())


def _yf_anchor_window():
    """Cierres S&P de yfinance cerca del ancla, solo para medir k cuando el df pedido no los trae."""
    import yfinance as yf
    d = yf.download("^MERV", start="2026-10-01", end="2026-10-17", progress=False, auto_adjust=False)
    if d.empty:
        return pd.Series(dtype=float)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    s = d["Close"].dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s


# ------------------------------------------------------------------ API
def empalmar_merval(df, start=None, end=None, price_col=None):
    """
    Devuelve `df` (^MERV en escala S&P) con las filas posteriores al 16/10/2026
    reemplazadas por MSCI Merval de Rava x k. Mantiene la forma del df recibido
    (índice de fechas o columna "Date"; devuelve solo la columna de precio).
    `start`/`end` (inclusivos) acotan el tramo nuevo al rango pedido. Si algo falla,
    devuelve df sin tocar.
    """
    try:
        had_date_col = df is not None and "Date" in getattr(df, "columns", [])
        work = df.copy() if df is not None else pd.DataFrame()
        if had_date_col:
            work = work.set_index("Date")
        if not work.empty:
            work.index = pd.to_datetime(work.index)
            if work.index.tz is not None:
                work.index = work.index.tz_localize(None)
            work.index = work.index.normalize()
            col = price_col or ("Close" if "Close" in work.columns else work.columns[-1])
        else:
            col = price_col or "Close"

        msci = get_msci_merval()
        if msci.empty:
            return df

        old_close = work[col].dropna() if not work.empty else pd.Series(dtype=float)
        k = _factor(old_close, msci) if not old_close.empty else None
        if k is None:  # el df no incluye días <= ancla (ej. start posterior): medirlo aparte
            k = _factor(_yf_anchor_window(), msci)
        if k is None:
            logger.warning("No se pudo calcular el factor S&P/MSCI; ^MERV queda sin empalmar")
            return df

        _K["v"] = k
        nuevo = (msci[msci.index > ANCHOR] * k).rename(col)
        if start is not None:
            nuevo = nuevo[nuevo.index >= pd.Timestamp(start)]
        elif not work.empty:
            nuevo = nuevo[nuevo.index >= work.index.min()]
        if end is not None:
            nuevo = nuevo[nuevo.index <= pd.Timestamp(end)]
        if work.empty:
            # sin datos viejos: solo se puede devolver el tramo nuevo
            out = nuevo.to_frame()
        else:
            viejo = work[work.index <= ANCHOR]
            base = viejo[[col]] if len(work.columns) > 1 else viejo
            out = pd.concat([base, nuevo.to_frame()]).sort_index()
            out = out[~out.index.duplicated(keep="last")]
        out.index.name = work.index.name or "Date"
        return out.reset_index() if had_date_col else out
    except Exception as e:
        logger.warning(f"empalmar_merval falló: {e}")
        return df


def a_escala_msci(df):
    """
    Paso FINAL (después de extender con Stooq): divide toda la serie ^MERV por k
    para llevarla a escala MSCI. No hace nada si ESCALA_MSCI es False o si no hubo
    empalme. Acepta índice de fechas o columna 'Date'.
    """
    try:
        k = _K["v"]
        if not ESCALA_MSCI or k is None or df is None or df.empty:
            return df
        out = df.copy()
        cols = [c for c in out.columns if c != "Date" and pd.api.types.is_numeric_dtype(out[c])]
        out[cols] = out[cols] / k
        return out
    except Exception as e:
        logger.warning(f"a_escala_msci falló: {e}")
        return df


# ------------------------------------------------------------------ reemplazo de yf.download
def _tickers_list(tickers):
    if isinstance(tickers, str):
        return [t for t in tickers.replace(",", " ").split() if t]
    return list(tickers)


def yf_download(tickers, *args, **kwargs):
    """
    Reemplazo directo de yf.download: igual salida, pero si entre los tickers está
    ^MERV, esa columna se empalma con el MSCI Merval (y se pasa a escala MSCI si
    ESCALA_MSCI). Si algo falla, devuelve lo que devolvió yfinance sin tocar.
    """
    import yfinance as yf
    data = yf.download(tickers, *args, **kwargs)
    try:
        tl = [t.upper() for t in _tickers_list(tickers)]
        if "^MERV" not in tl or data is None or data.empty:
            return data
        return _patch_download(data, tl, kwargs.get("start"), kwargs.get("end"))
    except Exception as e:
        logger.warning(f"yf_download: no se pudo empalmar ^MERV ({e})")
        return data


def _patch_download(data, tl, start, end):
    multi = isinstance(data.columns, pd.MultiIndex)
    idx = pd.to_datetime(data.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    data = data.copy()
    data.index = idx.normalize()

    if multi:
        fields = [f for f in data.columns.get_level_values(0).unique()
                  if (f, "^MERV") in data.columns]
        get = lambda f: data[(f, "^MERV")]
    else:  # un solo ticker, columnas planas
        fields = [f for f in data.columns]
        get = lambda f: data[f]
    price_f = next((f for f in ("Close", "Adj Close") if f in fields), None)
    if price_f is None:
        return data

    base = get(price_f).dropna().to_frame("Close")
    base.index.name = "Date"
    end_incl = None if end is None else pd.Timestamp(end) - pd.Timedelta(days=1)  # yfinance: end exclusivo
    emp = empalmar_merval(base, start, end_incl)
    if emp is base or emp is None or len(emp) == 0:
        return data
    emp = a_escala_msci(emp)["Close"]
    k = _K["v"] if ESCALA_MSCI else None

    new_index = data.index.union(emp.index).sort_values()
    data = data.reindex(new_index)
    for f in fields:
        col = ("%s" % f, "^MERV") if multi else f
        if f in ("Close", "Adj Close"):
            data[col] = emp.reindex(new_index)
        elif f in ("Open", "High", "Low") and k:
            vals = data[col].copy()
            vals[vals.index > ANCHOR] = float("nan")  # no hay OHL diario para el tramo nuevo
            data[col] = vals / k
    return data
