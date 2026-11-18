import argparse
import yfinance as yf
from merval_msci import yf_download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_rgb, to_hex
from datetime import datetime, timedelta, time as dtime
from scipy.interpolate import PchipInterpolator
from adjustText import adjust_text
from functools import lru_cache
import os
import warnings

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BG_COLOR = '#0f0f1a'
PAPER_COLOR = '#05050f'
TEXT_COLOR = '#e0e0ff'
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.25)

plt.style.use('dark_background')


# ===================== PALETA DISTINGUIBLE =====================
def _srgb_to_lab(rgb):
    """sRGB (0-1) -> CIELAB (D65). Acepta (3,) o (N,3)."""
    rgb = np.atleast_2d(rgb)
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]])
    xyz = (lin @ M.T) / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return np.stack([L, a, b], axis=1)


@lru_cache(maxsize=1)
def _build_palette(n_max=40, bg=BG_COLOR, min_L=55, min_bg_dist=40):
    """
    Farthest-point sampling en CIELAB: cada color nuevo es el más lejano
    (perceptualmente) de los ya elegidos. Los primeros N son siempre los N
    más distintos posibles. Se descartan colores oscuros o parecidos al fondo.
    """
    h, s, v = np.meshgrid(np.arange(0, 1, 1 / 72),
                          [0.55, 0.75, 0.95],
                          [0.80, 0.92, 1.0])
    cand = hsv_to_rgb(np.stack([h.ravel(), s.ravel(), v.ravel()], axis=1))
    lab = _srgb_to_lab(cand)
    bg_lab = _srgb_to_lab(np.array(to_rgb(bg)))[0]

    ok = (lab[:, 0] >= min_L) & (np.linalg.norm(lab - bg_lab, axis=1) >= min_bg_dist)
    cand, lab = cand[ok], lab[ok]

    chosen = [int(np.argmax(np.linalg.norm(lab - bg_lab, axis=1)))]
    min_d = np.linalg.norm(lab - lab[chosen[0]], axis=1)
    while len(chosen) < min(n_max, len(cand)):
        nxt = int(np.argmax(min_d))
        chosen.append(nxt)
        min_d = np.minimum(min_d, np.linalg.norm(lab - lab[nxt], axis=1))
    return [to_hex(cand[i]) for i in chosen]


def _label_fontsize(n_lines, max_size=18, min_size=9):
    """Tamaño de la etiqueta de cada ticker según cuántas líneas hay (pocas líneas -> más grande)."""
    return float(np.clip(21 - 0.55 * n_lines, min_size, max_size))


def _spread_labels(fig, ax, texts, min_gap_scale=1.2, relayout=True):
    """
    Separa verticalmente las etiquetas finales para que no se pisen, conservando el orden por altura
    y quedando lo más cerca posible de la posición real de cada línea. Mide en píxeles, así que
    funciona igual con 5 o con 40 etiquetas y con cualquier tamaño de letra.
    """
    if len(texts) < 2:
        return
    try:
        if relayout:
            fig.tight_layout()  # layout provisorio para conocer el alto real del eje
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        y0, y1 = ax.get_ylim()
        k = ax.get_window_extent(rend).height / (y1 - y0)      # píxeles por unidad de dato
        ys = np.array([t.get_position()[1] for t in texts], dtype=float)
        # Alto REAL de cada etiqueta (con la tipografía en uso) + margen; nunca menos que la estimación por tamaño
        est = np.array([t.get_fontsize() * fig.dpi / 72 * min_gap_scale for t in texts])
        real = np.array([t.get_window_extent(rend).height for t in texts]) + 0.25 * np.array(
            [t.get_fontsize() * fig.dpi / 72 for t in texts])
        hs = np.maximum(est, real)
        below = np.array([getattr(t, '_extra_below', 0.0) for t in texts])   # texto extra colgando debajo
        order = np.argsort(ys)
        pos = ys[order] * k
        up_ext = hs[order] / 2                       # hacia arriba de cada etiqueta
        dn_ext = hs[order] / 2 + below[order]        # hacia abajo (incluye el texto extra)
        need = up_ext[:-1] + dn_ext[1:]
        lo, hi = y0 * k + dn_ext[0], y1 * k - up_ext[-1]
        for _ in range(600):                                     # relajación: se empujan de a pares
            moved = False
            for i in range(len(pos) - 1):
                d = pos[i + 1] - pos[i]
                if d < need[i] - 0.5:
                    sft = (need[i] - d) / 2
                    pos[i] -= sft
                    pos[i + 1] += sft
                    moved = True
            pos = np.clip(pos, lo, hi)
            if not moved:
                break
        for i in range(len(pos) - 1):                            # pasada final: garantiza que no se solapen
            pos[i + 1] = max(pos[i + 1], pos[i] + need[i])
        if pos[-1] > hi:
            pos[-1] = hi
            for i in range(len(pos) - 2, -1, -1):
                pos[i] = min(pos[i], pos[i + 1] - need[i])
        for j, idx in enumerate(order):
            x = texts[idx].get_position()[0]
            texts[idx].set_position((x, pos[j] / k))
    except Exception as e:
        print(f"AVISO: no se pudieron separar las etiquetas ({e})")


def _avoid_title_overlap(fig, ax, pad_px=10):
    """Si el título (suptitle) pisa el eje, baja el borde superior del área de gráfico."""
    try:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        st = getattr(fig, '_suptitle', None)
        if st is None:
            return
        t_bottom = st.get_window_extent(r).y0
        a_top = ax.get_window_extent(r).y1
        if t_bottom - pad_px < a_top:
            need = (a_top - (t_bottom - pad_px)) / fig.bbox.height
            gs = ax.get_subplotspec().get_gridspec()
            gs.update(top=max(0.3, ax.get_position().y1 - need))
    except Exception as e:
        print(f"AVISO: no se pudo ajustar el título ({e})")


def get_line_colors(n):
    """Devuelve n colores lo más distintos posible entre sí."""
    pal = _build_palette()
    return [pal[i % len(pal)] for i in range(n)]


# ==================== DIVIDENDS ADJUSTMENTS ====================
DIVIDENDS = {
    'BYMA.BA': [
        (datetime(2026, 4, 14), 22.68)
    ]
}


def apply_dividend_adjustments(df):
    if df is None or df.empty:
        return df
    for ticker in df.columns:
        if ticker in DIVIDENDS:
            for ex_date, amount in DIVIDENDS[ticker]:
                mask = df.index <= ex_date
                df.loc[mask, ticker] -= amount
    return df


# ===================== HELPERS =====================
def find_closest_trading_day(df, target_date):
    if df.empty:
        return None
    target = pd.to_datetime(target_date)
    past_dates = df.index[df.index <= target]
    if not past_dates.empty:
        return past_dates.max()
    return df.index.min()


def _apply_extended_hours(df):
    """
    Reemplaza (o agrega) el último dato de cada ticker con el último precio de 1 minuto
    incluyendo Pre-Market y After-Hours. Solo tickers de EEUU (se omiten .BA e índices ^).
    Si la última sesión con datos es posterior a la última fila diaria, agrega esa fila.
    """
    cols = [c for c in df.columns if not str(c).upper().endswith('.BA') and not str(c).startswith('^')]
    if df.empty or not cols:
        return df
    try:
        intr = yf.download(cols, period='5d', interval='1m', prepost=True,
                           progress=False, auto_adjust=True)['Close']
        if isinstance(intr, pd.Series):
            intr = intr.to_frame(cols[0])
        intr = intr.dropna(how='all')
        if intr.empty:
            return df
        out = df.astype(float)  # copia de trabajo: si algo falla se devuelve df sin tocar
        last_daily = out.index[-1].normalize()
        for c in cols:
            if c not in intr.columns:
                continue
            sr = intr[c].dropna()
            if sr.empty:
                continue
            ts = sr.index[-1]
            ts = ts.tz_convert('America/New_York') if ts.tzinfo is not None \
                else ts.tz_localize('UTC').tz_convert('America/New_York')
            day = pd.Timestamp(ts.date())
            if day < last_daily:
                continue
            if day not in out.index:
                out.loc[day] = np.nan
                out = out.sort_index()
            out.loc[day, c] = float(sr.iloc[-1])
            print(f"DEBUG: {c} último precio con Pre/After Market ({ts:%Y-%m-%d %H:%M} NY) = {float(sr.iloc[-1]):.4f}")
        return out
    except Exception as e:
        print(f"AVISO: no se pudo usar Pre/After Market ({e})")
    return df


def fetch_data(tickers, start_date, extended_hours=False):
    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    data = yf_download(tickers, start=start_date, end=end_date,
                       progress=False, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all').dropna(axis=1, how='all')
    if extended_hours:
        data = _apply_extended_hours(data)
    data = apply_dividend_adjustments(data)
    data = data.ffill(limit=5).bfill(limit=5)
    return data


def fetch_ccl_proxy(start_date):
    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    ccl_data = yf.download(['YPFD.BA', 'YPF'], start=start_date, end=end_date,
                           progress=False, auto_adjust=True)['Close']
    if ccl_data.empty or 'YPFD.BA' not in ccl_data.columns or 'YPF' not in ccl_data.columns:
        return None
    ccl_ratio = ccl_data['YPFD.BA'] / ccl_data['YPF']
    ccl_ratio = ccl_ratio.interpolate(method='time').ffill().bfill()
    return ccl_ratio


def get_reference_date(period: str, now: datetime):
    if period == "YTD":
        return datetime(now.year, 1, 1) - timedelta(days=1)
    elif period == "MTD":
        if now.month == 1:
            return datetime(now.year - 1, 12, 31)
        else:
            return datetime(now.year, now.month, 1) - timedelta(days=1)
    elif period == "WTD":
        weekday = now.weekday()
        days_back = (weekday - 4) % 7 + 7
        if weekday == 4:
            days_back = 7
        return now - timedelta(days=days_back)
    else:
        return None


# ===================== INTRADAY FUNCTIONS =====================
def fetch_intraday_data(tickers, target_date=None, prepost=False):
    """Fetch 1-minute data for a specific date"""
    if target_date is None:
        target_date = datetime.now().date()

    target_dt = pd.to_datetime(target_date)
    start = target_dt - timedelta(days=3)
    end = target_dt + timedelta(days=2)

    data = yf.download(
        tickers, start=start, end=end, interval="1m",
        progress=False, prepost=prepost, auto_adjust=True
    )['Close']

    if isinstance(data, pd.Series):
        data = data.to_frame()

    if data.empty:
        return pd.DataFrame()

    data = data.dropna(how='all').dropna(axis=1, how='all')

    # Hora de Nueva York (con Pre/After Market el día en UTC se corta a las 20:00 NY)
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_convert('America/New_York')
    else:
        data.index = pd.to_datetime(data.index).tz_localize('UTC').tz_convert('America/New_York')

    target_date_only = target_dt.date()
    data = data[data.index.date == target_date_only]

    # En Pre/After Market hay minutos sin operaciones: se arrastra el último precio
    data = data.ffill(limit=None if prepost else 5).bfill(limit=5)
    return data


def _window_bounds(start_time, end_time):
    """(desde, hasta) como datetime.time, o None si no hay ventana."""
    if start_time is None and end_time is None:
        return None
    return (start_time or dtime(0, 0), end_time or dtime(23, 59, 59))


def _parse_ref_times(ref_times):
    """['09:30|Apertura', '14:00'] -> [(time(9,30), 'Apertura'), (time(14,0), '')]"""
    parsed = []
    for chunk in ref_times or []:
        chunk = str(chunk).strip()
        if not chunk:
            continue
        hhmm, _, label = chunk.partition("|")
        try:
            t = datetime.strptime(hhmm.strip(), "%H:%M").time()
        except ValueError:
            print(f"DEBUG: línea de referencia inválida ignorada: {chunk!r}")
            continue
        parsed.append((t, label.strip()))
    return parsed


def run_intraday_evolution(tickers, group_name="Activos", custom_title=None,
                           max_lines=15, smooth=True, baseline="open", target_date=None,
                           start_time=None, end_time=None, ref_times=None, prepost=False):
    df = fetch_intraday_data(tickers, target_date, prepost=prepost)
    if df.empty:
        return None
    return plot_intraday_evolution(df, group_name, custom_title, max_lines, smooth, baseline,
                                   start_time=start_time, end_time=end_time, ref_times=ref_times,
                                   prepost=prepost)


def plot_intraday_evolution(df, group_name, custom_title, max_lines=15, smooth=True, baseline="open",
                            start_time=None, end_time=None, ref_times=None, prepost=False):
    if df.empty or len(df) < 5:
        print("DEBUG: df intraday vacío o muy pequeño")
        return None

    # Con Pre/After Market se marcan apertura y cierre de la rueda regular
    if prepost:
        ref_times = list(ref_times or [])
        have = {t for t, _ in _parse_ref_times(ref_times)}
        for hhmm, label in (("09:30", "Apertura"), ("16:00", "Cierre")):
            if datetime.strptime(hhmm, "%H:%M").time() not in have:
                ref_times.append(f"{hhmm}|{label}")

    # Timezone to NY
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert('America/New_York')
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')

    print(f"DEBUG: Datos intraday para {len(df)} minutos, tickers: {list(df.columns)}")
    print(f"DEBUG: Rango de tiempo: {df.index[0]} → {df.index[-1]}")

    # Ventana horaria (hora de Nueva York, igual que el eje X)
    window = _window_bounds(start_time, end_time)
    if window:
        print(f"DEBUG: ventana horaria {window[0]} → {window[1]}")

    returns = pd.DataFrame(index=df.index)
    late_starters = []
    title_base = "desde Apertura"

    for col in df.columns:
        series = df[col].dropna()
        if series.empty or series.iloc[0] == 0:
            print(f"DEBUG: Ticker {col} skipped (empty or zero price)")
            continue

        if baseline == "open":
            market_open = series.index[0].replace(hour=9, minute=30, second=0, microsecond=0)
            candidates = series[series.index >= market_open]
            if not candidates.empty:
                ticker_base = float(candidates.iloc[0])
                if series.index[0] > market_open + pd.Timedelta(minutes=15):
                    late_starters.append(col)
            else:
                ticker_base = float(series.iloc[0])
                late_starters.append(col)
            title_base = "desde Apertura"
            print(f"DEBUG: {col} (Apertura) base = {ticker_base:.4f} at {series.index[0]}")
        elif baseline == "window":
            # Primer dato del ticker dentro de la ventana horaria
            s_win = series.between_time(window[0], window[1]) if window else series
            if s_win.empty:
                print(f"DEBUG: Ticker {col} skipped (sin datos en la ventana)")
                continue
            ticker_base = float(s_win.iloc[0])
            if window:
                title_base = f"desde Apertura de la ventana ({window[0]:%H:%M}–{window[1]:%H:%M})"
            else:
                title_base = "desde el primer dato del día"
            print(f"DEBUG: {col} (Ventana) base = {ticker_base:.4f} at {s_win.index[0]}")
        else:
            # Cierre anterior
            day_only = series.index[0].date()
            daily_end = pd.to_datetime(day_only)
            daily_start = daily_end - timedelta(days=10)
            print(f"DEBUG: {col} buscando previous close. daily_start={daily_start.date()}, daily_end={daily_end.date()}")
            daily = yf.download(col, start=daily_start, end=daily_end,
                                progress=False, auto_adjust=True)['Close']
            print(f"DEBUG: {col} daily data shape: {daily.shape}, dates: {list(daily.index.date) if not daily.empty else 'empty'}")
            if not daily.empty:
                prev_closes = daily[daily.index.date < day_only]
                if not prev_closes.empty:
                    ticker_base = float(prev_closes.iloc[-1].item() if hasattr(prev_closes.iloc[-1], 'item') else prev_closes.iloc[-1])
                    print(f"DEBUG: {col} previous close encontrado = {ticker_base:.4f} ({prev_closes.index[-1].date()})")
                else:
                    ticker_base = float(daily.iloc[0].item() if hasattr(daily.iloc[0], 'item') else daily.iloc[0])
                    print(f"DEBUG: {col} usando primer daily como fallback = {ticker_base:.4f}")
            else:
                ticker_base = float(series.iloc[0])
                print(f"DEBUG: {col} NO daily data → fallback a intraday first price = {ticker_base:.4f}")
            title_base = "desde Cierre Anterior"

        returns[col] = (series / ticker_base - 1) * 100

    returns = returns.dropna(axis=1, how='all')

    # Recorte horario de lo que se muestra (el baseline ya se calculó arriba)
    if window and not returns.empty:
        returns = returns.between_time(window[0], window[1]).dropna(how='all')
        if baseline != "window":
            title_base = f"{title_base} · ventana {window[0]:%H:%M}–{window[1]:%H:%M}"

    if returns.empty or returns.shape[1] == 0 or len(returns) < 2:
        print("DEBUG: returns quedó vacío después de limpiar. Ningún ticker válido.")
        return None

    if prepost:
        title_base = f"{title_base} · con Pre/After Market"

    if late_starters and baseline == "open":
        title_base = title_base.replace("desde Apertura", "desde Apertura (algunos tickers empezaron tarde)", 1)

    # Limit & sort
    final_returns = returns.iloc[-1].dropna()
    if len(returns.columns) > max_lines:
        top = list(final_returns.nlargest(max_lines // 2).index)
        bot = list(final_returns.nsmallest(max_lines // 2).index)
        selected = list(dict.fromkeys(top + bot))
        returns = returns[selected]

    sorted_cols = returns.iloc[-1].sort_values(ascending=False).index
    returns = returns[sorted_cols]

    # Paleta de colores distinguibles
    colors = get_line_colors(len(returns.columns))

    # === PLOTTING ===
    fig = plt.figure(figsize=(17, 9.5), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)

    n_points = len(returns.index)
    x_numeric = np.arange(n_points)
    line_ends = {}

    for i, ticker in enumerate(returns.columns):
        color = colors[i]
        y = returns[ticker].values
        line_ends[ticker] = (n_points - 1, y[-1])

        if smooth and n_points > 4:
            try:
                spline = PchipInterpolator(x_numeric, y)
                x_s = np.linspace(0, n_points - 1, 400)
                y_s = spline(x_s)
                ax.plot(x_s, y_s, color=color, linewidth=2.2, alpha=0.9)
            except:
                ax.plot(x_numeric, y, color=color, linewidth=2.2, alpha=0.9)
        else:
            ax.plot(x_numeric, y, color=color, linewidth=2.2, alpha=0.9)

    y_min = np.nanmin(returns.values)
    y_max = np.nanmax(returns.values)
    if not np.isfinite(y_min): y_min = -20
    if not np.isfinite(y_max): y_max = 20
    ax.set_ylim(y_min - abs(y_min) * 0.2, y_max + abs(y_max) * 0.2)

    ax.axhline(0, color='#ff4d4d', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.fill_between([-0.5, n_points * 2], 0, y_max + 100, color='green', alpha=0.02)
    ax.fill_between([-0.5, n_points * 2], 0, y_min - 100, color='red', alpha=0.02)

    label_x_pos = n_points + (n_points * 0.04)
    label_fs = _label_fontsize(len(returns.columns))
    texts = []
    for i, ticker in enumerate(returns.columns):
        color = colors[i]
        txt = ax.text(label_x_pos, returns[ticker].iloc[-1],
                      f" {ticker} ({returns[ticker].iloc[-1]:+.2f}%)",
                      color=color, fontsize=label_fs, fontweight='bold',
                      va='center', ha='left', zorder=5,
                      bbox=dict(facecolor=BG_COLOR, alpha=0.9, edgecolor='none', pad=1))
        texts.append(txt)

    _spread_labels(fig, ax, texts)

    if n_points > 10:
        t0 = returns.index[0]
        t1 = returns.index[-1]
        span_min = (t1 - t0).total_seconds() / 60
        tick_freq = ('5min' if span_min <= 60 else '10min' if span_min <= 120 else
                     '30min' if span_min <= 480 else '1h')
        tick_times = pd.date_range(start=t0.floor(tick_freq), end=t1.ceil(tick_freq), freq=tick_freq)
        tick_locs = []
        tick_labels = []
        for t in tick_times:
            deltas = np.abs((returns.index - t).total_seconds())
            idx = int(np.argmin(deltas))
            if idx not in tick_locs:
                tick_locs.append(idx)
                tick_labels.append(t.strftime('%H:%M'))
        ax.set_xticks(tick_locs)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    else:
        ax.set_xticks(range(0, n_points, max(1, n_points // 8)))
        ax.set_xticklabels([returns.index[i].strftime('%H:%M') for i in ax.get_xticks()], rotation=45, ha='right')

    ax.set_xlim(-0.5, n_points * 1.08)
    ax.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)

    # Líneas verticales de referencia (sutiles)
    for t_ref, label in _parse_ref_times(ref_times):
        ref_ts = returns.index[0].normalize().replace(hour=t_ref.hour, minute=t_ref.minute)
        if ref_ts < returns.index[0] or ref_ts > returns.index[-1]:
            print(f"DEBUG: referencia {t_ref:%H:%M} fuera del rango mostrado, se omite")
            continue
        idx = int(np.argmin(np.abs((returns.index - ref_ts).total_seconds())))
        ax.axvline(x=idx, color='#9a9ac0', linestyle=':', linewidth=1.0, alpha=0.45, zorder=2)
        ax.text(idx, 0.985, f" {label or t_ref.strftime('%H:%M')} ",
                transform=ax.get_xaxis_transform(), rotation=90, va='top', ha='right',
                fontsize=10, color='#b4b4d8', alpha=0.8, zorder=3,
                bbox=dict(facecolor=BG_COLOR, alpha=0.6, edgecolor='none', pad=1.5))

    title = custom_title or group_name
    fig.suptitle(f"Evolución Intradía (NY Time) – {title_base}\n{title}",
                 fontsize=22, color=TEXT_COLOR, fontweight='bold', y=0.96)

    fig.text(0.5, 0.5, "@MTaurus_ok", fontsize=90, color=(140 / 255, 140 / 255, 180 / 255, 0.09),
             ha='center', va='center', rotation=30, zorder=999)

    plt.tight_layout()
    _avoid_title_overlap(fig, ax)
    output_path = "intraday_evolution.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return output_path


# ===================== ORIGINAL PLOT & RUNNER =====================
EVENT_COLOR = '#ffd166'


def _norm_list(x):
    """Acepta 'A, B' o ['A','B'] y devuelve ['A','B'] en mayúsculas."""
    if not x:
        return []
    if isinstance(x, str):
        x = x.split(",")
    return [str(t).strip().upper() for t in x if str(t).strip()]


def _text_color_for(bg):
    r, g, b = to_rgb(bg)
    return '#05050f' if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else '#ffffff'


def _draw_rank_strip(a, who, color_of, label):
    """Franja fina: cada día se pinta con el color del ticker (líder o último). Los rótulos se agregan al final."""
    n = len(who)
    x = np.arange(n)
    a.bar(x, 1, width=1.0, color=[color_of.get(t, (0, 0, 0, 0)) for t in who], align='center', linewidth=0)
    a.set_ylim(0, 1)
    a.set_yticks([])
    for sp in a.spines.values():
        sp.set_visible(False)
    a.text(-0.008, 0.5, label, transform=a.transAxes, ha='right', va='center',
           fontsize=11, color=TEXT_COLOR, fontweight='bold')


def _label_strip_runs(fig, a, who, color_of):
    """
    Escribe el ticker dentro de cada tramo de la franja si entra en el ancho disponible (en píxeles),
    achicando la letra hasta 7 pt. Se llama con el layout final, así que sirve con 10 o con 250 ruedas.
    """
    from itertools import groupby
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    x0, x1 = a.get_xlim()
    ext = a.get_window_extent(r)
    px_per_x = ext.width / (x1 - x0)
    max_fs = min(11.0, 0.62 * ext.height * 72 / fig.dpi)
    pos = 0
    for t, grp in groupby(who):
        length = len(list(grp))
        if t in color_of:
            avail = length * px_per_x * 0.94
            fs = max_fs
            while fs >= 7:
                txt = a.text(pos + (length - 1) / 2, 0.5, t, ha='center', va='center', fontsize=fs,
                             fontweight='bold', color=_text_color_for(color_of[t]), clip_on=True)
                if txt.get_window_extent(r).width <= avail:
                    break
                txt.remove()
                fs -= 1
        pos += length


def plot_returns_evolution(df, group_name, custom_title, period="YTD",
                           max_lines=20, smooth=True, frequency='D',
                           highlight=None, mark_extremes=False, rank_strips=False, events=None):
    if df.empty or len(df) < 2:
        return None

    freq_map = {
        'D': None,
        'W': 'W-FRI',
        'M': 'ME'
    }
    resample_freq = freq_map.get(frequency.upper())
    if resample_freq:
        df = df.resample(resample_freq).last()
        df = df.dropna(how='all')
        df = df.ffill()

    returns = (df / df.iloc[0] - 1) * 100
    if returns.empty or len(returns) < 2:
        return None

    final_returns = returns.iloc[-1]
    hl_req = _norm_list(highlight)
    if len(returns.columns) > max_lines:
        top = list(final_returns.nlargest(max_lines // 2).index)
        bot = list(final_returns.nsmallest(max_lines // 2).index)
        selected = list(dict.fromkeys(top + bot + [t for t in hl_req if t in returns.columns]))
        returns = returns[selected]

    sorted_cols = returns.iloc[-1].sort_values(ascending=False).index
    returns = returns[sorted_cols]

    # Paleta de colores distinguibles
    colors = get_line_colors(len(returns.columns))
    color_of = {t: colors[i] for i, t in enumerate(returns.columns)}
    hl = [t for t in hl_req if t in returns.columns]
    for t in hl_req:
        if t not in returns.columns:
            print(f"AVISO: '{t}' no está en el gráfico; no se puede destacar")

    fig = plt.figure(figsize=(17, 9.5 + (1.1 if rank_strips else 0)), facecolor=PAPER_COLOR)
    if rank_strips:
        gs = fig.add_gridspec(3, 1, height_ratios=[22, 0.9, 0.9], hspace=0.12)
        ax = fig.add_subplot(gs[0], facecolor=BG_COLOR)
        ax_lead = fig.add_subplot(gs[1], sharex=ax, facecolor=BG_COLOR)
        ax_last = fig.add_subplot(gs[2], sharex=ax, facecolor=BG_COLOR)
    else:
        ax = fig.add_subplot(111, facecolor=BG_COLOR)

    n_points = len(returns.index)
    x_numeric = np.arange(n_points)
    line_ends = {}

    def _draw(y, color, lw, alpha, z):
        if smooth and n_points > 4:
            try:
                spline = PchipInterpolator(x_numeric, y)
                x_s = np.linspace(0, n_points - 1, 400)
                ax.plot(x_s, spline(x_s), color=color, linewidth=lw, alpha=alpha, zorder=z)
                return
            except Exception:
                pass
        ax.plot(x_numeric, y, color=color, linewidth=lw, alpha=alpha, zorder=z)

    for i, ticker in enumerate(returns.columns):
        color = colors[i]
        y = returns[ticker].values
        line_ends[ticker] = (n_points - 1, y[-1])
        if hl:
            if ticker in hl:
                _draw(y, color, 4.2, 1.0, 7)      # destacado
            else:
                _draw(y, color, 1.5, 0.28, 3)     # atenuado
        else:
            _draw(y, color, 2.2, 0.9, 4)

    years = returns.index.year
    year_changes = np.where(years[1:] != years[:-1])[0] + 1
    year_texts = []
    for idx in year_changes:
        year = returns.index[idx].year
        x_pos = idx
        window = 8
        start = max(0, idx - window)
        end = min(len(returns), idx + window + 1)
        local_max = returns.iloc[start:end].max().max()
        y_pos = local_max * 1.09
        y_max_plot = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100
        if y_pos > y_max_plot * 0.94:
            y_pos = y_max_plot * 0.88
        txt = ax.text(x_pos, y_pos, str(year),
                      rotation=90, va='bottom', ha='center', fontsize=11,
                      color='#a0a0cc', fontweight='bold', alpha=0.9,
                      bbox=dict(facecolor=BG_COLOR, alpha=0.75, edgecolor='none', pad=3))
        year_texts.append(txt)
        ax.axvline(x=idx, color='#666688', linestyle='--', linewidth=1.1, alpha=0.55, zorder=2)

    if year_texts:
        try:
            adjust_text(year_texts, ax=ax,
                        only_move={'texts': 'y'},
                        expand_text=(1.15, 1.4),
                        force_text=(0.1, 1.0),
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5, alpha=0.3))
        except Exception:
            pass

    y_min, y_max = returns.min().min(), returns.max().max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

    ax.fill_between([-0.5, n_points * 2], 0, y_max + 100, color='green', alpha=0.02)
    ax.fill_between([-0.5, n_points * 2], 0, y_min - 100, color='red', alpha=0.02)
    ax.axhline(0, color='#ff4d4d', linewidth=1.5, linestyle='--', alpha=0.8)

    ax.set_xlim(-0.5, n_points * 1.35)

    # ---- Pico / valle / último dato de los tickers destacados ----
    peak_info = {}
    if hl and mark_extremes:
        last_i = n_points - 1
        for t in hl:
            yv = returns[t].values
            c = color_of[t]
            i_max, i_min = int(np.nanargmax(yv)), int(np.nanargmin(yv))
            peak_info[t] = (i_max, yv[i_max])
            ax.scatter([last_i], [yv[-1]], s=90, color=c, edgecolor='white', linewidth=1.3, zorder=9)
            bbox = dict(facecolor=BG_COLOR, alpha=0.88, edgecolor=c, boxstyle='round,pad=0.3', linewidth=1)
            if i_max not in (0, last_i):
                ax.scatter([i_max], [yv[i_max]], s=110, marker='^', color=c, edgecolor='white', linewidth=1.2, zorder=9)
                near_r, near_l = i_max > 0.88 * n_points, i_max < 0.08 * n_points
                ax.annotate(f"{t} · pico {yv[i_max]:+.1f}%\n{returns.index[i_max]:%d/%m/%Y}",
                            xy=(i_max, yv[i_max]), xytext=(-14 if near_r else 14 if near_l else 0, 20),
                            textcoords='offset points',
                            ha='right' if near_r else 'left' if near_l else 'center', va='bottom',
                            fontsize=12, color=c, fontweight='bold',
                            bbox=bbox, zorder=10,
                            arrowprops=dict(arrowstyle='-', color=c, lw=1, alpha=0.8))
            if i_min not in (0, last_i):
                ax.scatter([i_min], [yv[i_min]], s=110, marker='v', color=c, edgecolor='white', linewidth=1.2, zorder=9)
                near_r, near_l = i_min > 0.88 * n_points, i_min < 0.08 * n_points
                ax.annotate(f"{t} · mín. {yv[i_min]:+.1f}%\n{returns.index[i_min]:%d/%m/%Y}",
                            xy=(i_min, yv[i_min]), xytext=(-14 if near_r else 14 if near_l else 0, -20),
                            textcoords='offset points',
                            ha='right' if near_r else 'left' if near_l else 'center', va='top',
                            fontsize=12, color=c, fontweight='bold',
                            bbox=bbox, zorder=10,
                            arrowprops=dict(arrowstyle='-', color=c, lw=1, alpha=0.8))

    # ---- Eventos (líneas verticales con texto) ----
    for ev in (events or []):
        try:
            d_txt, label_ev = str(ev).split('|', 1)
            dt = pd.to_datetime(d_txt.strip())
        except Exception:
            print(f"AVISO: evento ignorado (formato 'YYYY-MM-DD|texto'): {ev!r}")
            continue
        if dt < returns.index[0] or dt > returns.index[-1]:
            continue
        ei = int(min(returns.index.searchsorted(dt), n_points - 1))
        ax.axvline(ei, color=EVENT_COLOR, linestyle=(0, (3, 3)), linewidth=1.3, alpha=0.85, zorder=3)
        ax.text(ei - 0.3, ax.get_ylim()[1] * 0.985 if ax.get_ylim()[1] > 0 else ax.get_ylim()[1],
                f"{dt:%d/%m} · {label_ev.strip()}", rotation=90, va='top', ha='right', fontsize=11,
                color=EVENT_COLOR, fontweight='bold', zorder=6,
                bbox=dict(facecolor=BG_COLOR, alpha=0.8, edgecolor='none', pad=2))

    label_x_pos = n_points + (n_points * 0.05)
    label_fs = _label_fontsize(len(returns.columns))
    texts = []
    for i, ticker in enumerate(returns.columns):
        color = colors[i]
        is_hl = ticker in hl
        lbl = f" {ticker} ({final_returns[ticker]:+.1f}%)"
        txt = ax.text(label_x_pos, returns[ticker].iloc[-1], lbl,
                      color=color, fontsize=label_fs * (1.15 if is_hl else 1.0), fontweight='bold',
                      va='center', ha='left', zorder=5,
                      bbox=dict(facecolor=BG_COLOR, alpha=0.9, edgecolor='none', pad=1))
        if hl and not is_hl:
            txt.set_alpha(0.6)
        if is_hl and ticker in peak_info and peak_info[ticker][0] != n_points - 1:
            txt._extra_below = label_fs * 1.75 * fig.dpi / 72   # lugar para "pp vs pico" debajo
        txt.set_gid(ticker)
        texts.append(txt)

    # ---- Franjas de ranking: quién lideraba / quién iba último cada día ----
    if rank_strips:
        lead = list(returns.idxmax(axis=1))
        last = list(returns.idxmin(axis=1))
        if (returns.iloc[0].abs() < 1e-9).all():   # día base: todos en 0, sin ranking
            lead[0] = last[0] = None
        _draw_rank_strip(ax_lead, lead, color_of, "1°")
        _draw_rank_strip(ax_last, last, color_of, "Último")

    tick_ax = ax_last if rank_strips else ax
    tick_ax.set_xticks(range(0, n_points, max(1, n_points // 12)))
    tick_ax.set_xticklabels([returns.index[i].strftime('%b %d') for i in tick_ax.get_xticks()],
                            rotation=45, ha='right')
    if rank_strips:
        ax.tick_params(labelbottom=False)
        ax_lead.tick_params(labelbottom=False, length=0)

    ax.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)

    freq_names = {'D': 'Diaria', 'W': 'Semanal', 'M': 'Mensual'}
    freq_text = freq_names.get(frequency.upper(), 'Diaria')
    title_period = f"{period} Evolution ({freq_text})"

    fig.suptitle(f"Evolución de Retornos Acumulados\n{title_period} — {custom_title or group_name}",
                 fontsize=22, color=TEXT_COLOR, fontweight='bold', y=0.96)

    fig.text(0.5, 0.5, "@MTaurus_ok", fontsize=90, color=(140 / 255, 140 / 255, 180 / 255, 0.09),
             ha='center', va='center', rotation=30, zorder=999)

    plt.tight_layout()
    _avoid_title_overlap(fig, ax)
    # Con el layout ya definitivo se separan las etiquetas (medidas en píxeles reales) y se dibujan los conectores
    _spread_labels(fig, ax, texts, relayout=False)

    for txt in texts:
        ticker = txt.get_gid()
        target_x, target_y = line_ends[ticker]
        label_x, label_y = txt.get_position()
        color = txt.get_color()
        if ticker in hl and ticker in peak_info and peak_info[ticker][0] != n_points - 1:
            drop = returns[ticker].iloc[-1] - peak_info[ticker][1]
            ax.annotate(f"{drop:+.1f} pp vs pico", xy=(label_x, label_y), xycoords='data',
                        xytext=(6, -(label_fs * 1.15 * 0.95)), textcoords='offset points',
                        ha='left', va='top', fontsize=max(10, label_fs * 0.72), fontweight='bold',
                        color=color, zorder=6)
        ax.annotate('',
                    xy=(target_x, target_y),
                    xytext=(label_x - 0.5, label_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8,
                                    alpha=0.8 if ticker in hl else 0.4,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0"),
                    zorder=3)

    if rank_strips:
        _label_strip_runs(fig, ax_lead, lead, color_of)
        _label_strip_runs(fig, ax_last, last, color_of)
    output_path = "returns_evolution.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return output_path


def run_returns_evolution(
        tickers,
        period="YTD",
        start_date=None,
        group_name="Activos",
        custom_title=None,
        max_lines=20,
        exclude_tickers=None,
        smooth=True,
        ccl_adjust=False,
        frequency='D',
        custom_start_date=None,
        extended_hours=False,
        highlight=None,
        mark_extremes=False,
        rank_strips=False,
        events=None
):
    now = datetime.now()

    if not start_date:
        period_upper = period.upper()
        if period_upper == "YTD":
            start_date = f"{now.year - 1}-12-01"
        elif period_upper in ["MTD", "WTD"]:
            start_date = (now - timedelta(days=90)).strftime("%Y-%m-%d")
        else:
            days = {"1Y": 365, "2Y": 730, "5Y": 1825}.get(period_upper, 365)
            start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    fetch_start = start_date
    if custom_start_date:
        fetch_start = (datetime.combine(custom_start_date, datetime.min.time()) - timedelta(days=30)).strftime("%Y-%m-%d")

    df = fetch_data(tickers, fetch_start, extended_hours=extended_hours)

    if exclude_tickers:
        exclude_list = [t.strip().upper() for t in exclude_tickers.split(",") if t.strip()]
        df = df.drop(columns=[c for c in exclude_list if c in df.columns], errors='ignore')

    if df.empty:
        return None

    df_for_plot = df.copy()

    if custom_start_date:
        baseline_idx = find_closest_trading_day(df, custom_start_date)
        if baseline_idx is not None:
            df_for_plot = df.loc[baseline_idx:].copy()
            if custom_title:
                custom_title = f"{custom_title} (desde {baseline_idx.date()})"
            else:
                custom_title = f"desde {baseline_idx.date()}"
    else:
        ref_date = get_reference_date(period.upper(), now)
        if ref_date is not None:
            mask = df.index <= ref_date
            if mask.any():
                baseline_idx = df.index[mask].max()
                df_for_plot = df.loc[baseline_idx:].copy()

    if ccl_adjust:
        ccl_ratio = fetch_ccl_proxy(fetch_start)
        if ccl_ratio is not None:
            df_for_plot, ccl_ratio_aligned = df_for_plot.align(ccl_ratio, axis=0, join='inner')
            df_for_plot = df_for_plot.div(ccl_ratio_aligned, axis=0)
            custom_title = (custom_title + " (Ajustado CCL)" if custom_title else "Ajustado CCL")
        else:
            print("Warning: Could not fetch CCL proxy correctly.")

    if extended_hours:
        custom_title = f"{custom_title or group_name} · último dato con Pre/After Market"

    return plot_returns_evolution(
        df_for_plot,
        group_name,
        custom_title,
        period=period.upper(),
        max_lines=max_lines,
        smooth=smooth,
        frequency=frequency,
        highlight=highlight,
        mark_extremes=mark_extremes,
        rank_strips=rank_strips,
        events=events
    )


# ===================== CLI =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tickers', nargs='+', required=True)
    parser.add_argument('--period', type=str, default='YTD')
    parser.add_argument('--group_name', type=str, default="Activos")
    parser.add_argument('--custom_title', type=str, default=None)
    parser.add_argument('--max_lines', type=int, default=20)
    parser.add_argument('--exclude_tickers', type=str, default=None)
    parser.add_argument('--smooth', type=str, default='s')
    parser.add_argument('--ccl_adjust', action='store_true')
    parser.add_argument('--extended_hours', action='store_true',
                        help='Usar el último precio incluyendo Pre/After Market (solo EEUU)')
    parser.add_argument('--highlight', nargs='+', default=None,
                        help='Tickers a destacar (el resto se atenúa)')
    parser.add_argument('--mark_extremes', action='store_true',
                        help='Marcar pico, mínimo y último dato de los tickers destacados')
    parser.add_argument('--rank_strips', action='store_true',
                        help='Franjas debajo: quién lideraba y quién iba último cada día')
    parser.add_argument('--events', nargs='*', default=None,
                        help="Eventos 'YYYY-MM-DD|texto' (líneas verticales con etiqueta)")
    args = parser.parse_args()

    is_smooth = args.smooth.lower() in ['s', 'true', '1', 'yes']

    path = run_returns_evolution(
        args.tickers, args.period, None, args.group_name,
        args.custom_title, args.max_lines, args.exclude_tickers, is_smooth, args.ccl_adjust,
        extended_hours=args.extended_hours,
        highlight=args.highlight, mark_extremes=args.mark_extremes,
        rank_strips=args.rank_strips, events=args.events
    )
    if path:
        print(os.path.abspath(path))
