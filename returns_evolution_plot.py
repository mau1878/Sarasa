import argparse
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.interpolate import make_interp_spline
from adjustText import adjust_text
import os
import warnings
warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
BG_COLOR = '#0f0f1a'
PAPER_COLOR = '#05050f'
TEXT_COLOR = '#e0e0ff'
GRID_COLOR = (120 / 255, 120 / 255, 140 / 255, 0.25)

LINE_COLORS = [
    '#ff7070', '#40c0ff', '#00e673', '#ffd700', '#ba55d3',
    '#32cd32', '#1e90ff', '#ff8c00', '#ff69b4', '#7fff00',
    '#ff1493', '#00ffff', '#adff2f', '#ff4500', '#9370db',
    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeead',
    '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43',
    '#ee5253', '#10ac84', '#0abde3', '#f368e0', '#3f72af',
    '#ff4757', '#3742fa', '#2ed573', '#ffa502', '#c44569'
]
plt.style.use('dark_background')

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

def fetch_data(tickers, start_date):
    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=start_date, end=end_date,
                       progress=False, auto_adjust=True)['Close']
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how='all').dropna(axis=1, how='all')
    data = apply_dividend_adjustments(data)
    data = data.ffill(limit=5).bfill(limit=5)
    return data

def fetch_ccl_proxy(start_date):
    end_date = (datetime.now() + timedelta(days=2)).strftime('%Y-%m-%d')
    ccl_data = yf.download(['YPFD.BA', 'YPF'], start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
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
# ===================== INTRADAY FUNCTIONS =====================
# ===================== IMPROVED INTRADAY FUNCTIONS =====================
# ===================== IMPROVED INTRADAY FUNCTIONS =====================
def fetch_intraday_data(tickers, target_date=None):
    """Fetch 1-minute data for a specific date"""
    if target_date is None:
        target_date = datetime.now().date()
    
    target_dt = pd.to_datetime(target_date)
    start = target_dt - timedelta(days=3)
    end = target_dt + timedelta(days=2)
    
    data = yf.download(
        tickers, start=start, end=end, interval="1m",
        progress=False, prepost=False, auto_adjust=True
    )['Close']
    
    if isinstance(data, pd.Series):
        data = data.to_frame()
    if data.empty:
        return pd.DataFrame()
    
    data = data.dropna(how='all').dropna(axis=1, how='all')
    
    # Robust timezone handling
    if hasattr(data.index, 'tz') and data.index.tz is not None:
        data.index = data.index.tz_convert(None)
    
    target_date_only = target_dt.date()
    data = data[data.index.date == target_date_only]
    
    data = data.ffill(limit=5).bfill(limit=5)
    return data


def run_intraday_evolution(tickers, group_name="Activos", custom_title=None,
                          max_lines=15, smooth=True, baseline="open", target_date=None):
    df = fetch_intraday_data(tickers, target_date)
    if df.empty:
        return None
    return plot_intraday_evolution(df, group_name, custom_title, max_lines, smooth, baseline)


def plot_intraday_evolution(df, group_name, custom_title, max_lines=15, smooth=True, baseline="open"):
    if df.empty or len(df) < 5:
        print("DEBUG: df intraday vacío o muy pequeño")
        return None

    # Timezone to NY
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert('America/New_York')
    else:
        df = df.copy()
        df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('America/New_York')

    print(f"DEBUG: Datos intraday para {len(df)} minutos, tickers: {list(df.columns)}")
    print(f"DEBUG: Rango de tiempo: {df.index[0]} → {df.index[-1]}")

    returns = pd.DataFrame(index=df.index)
    late_starters = []

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
        else:
            # Cierre anterior
            target_date = series.index[0].date()
            daily_end = pd.to_datetime(target_date)
            daily_start = daily_end - timedelta(days=10)
            
            print(f"DEBUG: {col} buscando previous close. daily_start={daily_start.date()}, daily_end={daily_end.date()}")
            
            daily = yf.download(col, start=daily_start, end=daily_end, 
                              progress=False, auto_adjust=True)['Close']
            
            print(f"DEBUG: {col} daily data shape: {daily.shape}, dates: {list(daily.index.date) if not daily.empty else 'empty'}")
            
            if not daily.empty:
                prev_closes = daily[daily.index.date < target_date]
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
    if returns.empty or returns.shape[1] == 0:
        print("DEBUG: returns quedó vacío después de limpiar. Ningún ticker válido.")
        return None

    if late_starters and baseline == "open":
        title_base = "desde Apertura (algunos tickers empezaron tarde)"

    # Limit & sort
    final_returns = returns.iloc[-1].dropna()
    if len(returns.columns) > max_lines:
        top = list(final_returns.nlargest(max_lines//2).index)
        bot = list(final_returns.nsmallest(max_lines//2).index)
        selected = list(dict.fromkeys(top + bot))
        returns = returns[selected]

    sorted_cols = returns.iloc[-1].sort_values(ascending=False).index
    returns = returns[sorted_cols]

    # === PLOTTING ===
    fig = plt.figure(figsize=(17, 9.5), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)
    n_points = len(returns.index)
    x_numeric = np.arange(n_points)
    line_ends = {}

    for i, ticker in enumerate(returns.columns):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        y = returns[ticker].values
        line_ends[ticker] = (n_points - 1, y[-1])

        if smooth and n_points > 4:
            try:
                spline = make_interp_spline(x_numeric, y, k=2)
                x_s = np.linspace(0, n_points-1, 400)
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

    ax.set_ylim(y_min - abs(y_min)*0.2, y_max + abs(y_max)*0.2)
    ax.axhline(0, color='#ff4d4d', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.fill_between([-0.5, n_points*2], 0, y_max+100, color='green', alpha=0.02)
    ax.fill_between([-0.5, n_points*2], 0, y_min-100, color='red', alpha=0.02)

    label_x_pos = n_points + (n_points * 0.04)
    texts = []
    for i, ticker in enumerate(returns.columns):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        txt = ax.text(label_x_pos, returns[ticker].iloc[-1],
                      f" {ticker} ({returns[ticker].iloc[-1]:+.2f}%)",
                      color=color, fontsize=10, fontweight='bold',
                      va='center', ha='left', zorder=5,
                      bbox=dict(facecolor=BG_COLOR, alpha=0.9, edgecolor='none', pad=1))
        texts.append(txt)

    try:
        adjust_text(texts, ax=ax, only_move={'texts': 'y'}, autoalign='y',
                    expand_text=(1.2, 2.2), force_text=(0, 2.5))
    except:
        pass

    if n_points > 10:
        start_time = returns.index[0]
        end_time = returns.index[-1]
        tick_times = pd.date_range(start=start_time.floor('30min'), end=end_time.ceil('30min'), freq='30min')
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
        ax.set_xticks(range(0, n_points, max(1, n_points//8)))
        ax.set_xticklabels([returns.index[i].strftime('%H:%M') for i in ax.get_xticks()], rotation=45, ha='right')

    ax.set_xlim(-0.5, n_points * 1.08)
    ax.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)

    title = custom_title or group_name
    fig.suptitle(f"Evolución Intradía (NY Time) – {title_base}\n{title}",
                 fontsize=22, color=TEXT_COLOR, fontweight='bold', y=0.96)

    fig.text(0.5, 0.5, "@MTaurus_ok", fontsize=90, color=(140/255,140/255,180/255,0.09),
             ha='center', va='center', rotation=30, zorder=999)

    plt.tight_layout()
    output_path = "intraday_evolution.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=PAPER_COLOR)
    plt.close(fig)
    return output_path


def run_intraday_evolution(tickers, group_name="Activos", custom_title=None,
                          max_lines=15, smooth=True, baseline="open", target_date=None):
    df = fetch_intraday_data(tickers, target_date)
    if df.empty:
        return None
    return plot_intraday_evolution(df, group_name, custom_title, max_lines, smooth, baseline)

# ===================== ORIGINAL PLOT & RUNNER =====================
def plot_returns_evolution(df, group_name, custom_title, period="YTD",
                          max_lines=20, smooth=True, frequency='D'):
   
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
   
    if len(returns.columns) > max_lines:
        top = list(final_returns.nlargest(max_lines // 2).index)
        bot = list(final_returns.nsmallest(max_lines // 2).index)
        selected = list(dict.fromkeys(top + bot))
        returns = returns[selected]
   
    sorted_cols = returns.iloc[-1].sort_values(ascending=False).index
    returns = returns[sorted_cols]
   
    fig = plt.figure(figsize=(17, 9.5), facecolor=PAPER_COLOR)
    ax = fig.add_subplot(111, facecolor=BG_COLOR)
    n_points = len(returns.index)
    x_numeric = np.arange(n_points)
    line_ends = {}
   
    for i, ticker in enumerate(returns.columns):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        y = returns[ticker].values
        line_ends[ticker] = (n_points - 1, y[-1])
       
        if smooth and n_points > 4:
            try:
                spline = make_interp_spline(x_numeric, y, k=2)
                x_s = np.linspace(0, n_points - 1, 400)
                y_s = spline(x_s)
                ax.plot(x_s, y_s, color=color, linewidth=2.2, alpha=0.9, zorder=4)
            except:
                ax.plot(x_numeric, y, color=color, linewidth=2.2, alpha=0.9, zorder=4)
        else:
            ax.plot(x_numeric, y, color=color, linewidth=2.2, alpha=0.9, zorder=4)
   
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
        except:
            pass
   
    y_min, y_max = returns.min().min(), returns.max().max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)
    ax.fill_between([-0.5, n_points * 2], 0, y_max + 100, color='green', alpha=0.02)
    ax.fill_between([-0.5, n_points * 2], 0, y_min - 100, color='red', alpha=0.02)
    ax.axhline(0, color='#ff4d4d', linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_xlim(-0.5, n_points * 1.35)
   
    label_x_pos = n_points + (n_points * 0.05)
    texts = []
    for i, ticker in enumerate(returns.columns):
        color = LINE_COLORS[i % len(LINE_COLORS)]
        txt = ax.text(label_x_pos, returns[ticker].iloc[-1],
                      f" {ticker} ({final_returns[ticker]:+.1f}%)",
                      color=color, fontsize=10, fontweight='bold',
                      va='center', ha='left', zorder=5,
                      bbox=dict(facecolor=BG_COLOR, alpha=0.9, edgecolor='none', pad=1))
        txt.set_gid(ticker)
        texts.append(txt)
   
    try:
        adjust_text(texts, ax=ax, only_move={'texts': 'y'}, autoalign='y',
                    expand_text=(1.2, 2.2), force_text=(0, 2.5))
    except:
        pass
   
    for txt in texts:
        ticker = txt.get_gid()
        target_x, target_y = line_ends[ticker]
        label_x, label_y = txt.get_position()
        color = txt.get_color()
        ax.annotate('',
                    xy=(target_x, target_y),
                    xytext=(label_x - 0.5, label_y),
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8, alpha=0.4,
                                    shrinkA=0, shrinkB=0, connectionstyle="arc3,rad=0"),
                    zorder=3)
   
    ax.set_xticks(range(0, n_points, max(1, n_points // 12)))
    ax.set_xticklabels([returns.index[i].strftime('%b %d') for i in ax.get_xticks()],
                       rotation=45, ha='right')
    ax.grid(True, color=GRID_COLOR, linestyle=':', linewidth=0.5, alpha=0.3)
   
    freq_names = {'D': 'Diaria', 'W': 'Semanal', 'M': 'Mensual'}
    freq_text = freq_names.get(frequency.upper(), 'Diaria')
    title_period = f"{period} Evolution ({freq_text})"
    fig.suptitle(f"Evolución de Retornos Acumulados\n{title_period} — {custom_title or group_name}",
                 fontsize=22, color=TEXT_COLOR, fontweight='bold', y=0.96)
   
    fig.text(0.5, 0.5, "@MTaurus_ok", fontsize=90, color=(140/255,140/255,180/255,0.09),
             ha='center', va='center', rotation=30, zorder=999)
   
    plt.tight_layout()
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
    custom_start_date=None
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
   
    df = fetch_data(tickers, fetch_start)
   
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
   
    return plot_returns_evolution(
        df_for_plot,
        group_name,
        custom_title,
        period=period.upper(),
        max_lines=max_lines,
        smooth=smooth,
        frequency=frequency
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
    args = parser.parse_args()
    is_smooth = args.smooth.lower() in ['s', 'true', '1', 'yes']
    path = run_returns_evolution(
        args.tickers, args.period, None, args.group_name,
        args.custom_title, args.max_lines, args.exclude_tickers, is_smooth, args.ccl_adjust
    )
    if path:
        print(os.path.abspath(path))
